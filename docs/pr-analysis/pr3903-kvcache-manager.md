# PR #3903: KVCacheManager — Standalone Host-Side KV Cache

- **Author**: jijoongmoon
- **Date**: 2026-04-27
- **Branch**: `feature/kvcache-manager` → `main`
- **Size**: +12,120 / -2,820 lines (100 files)
- **Status**: OPEN (Wait 4 #3902)
- **Depends on**: #3902 (CausalLM Tensor API 마이그레이션)

---

## 왜 필요한가?

기존 CausalLM의 KV Cache는 **각 mha_core 레이어 내부에서 개별적으로 소유**하는 `Tensor`였다. 이 방식의 문제점:

1. **호스트 측에서 접근 불가**: KV Cache를 저장/로드/제어하려면 `model->forEachLayer()`로 모든 mha_core를 순회하며 내부 텐서를 찾아서 조작해야 함
2. **메모리 레이아웃 통제 불가**: 각 mha_core가 독립적으로 할당 → DSP/NPU 공유 메모리로의 통합 마이그레이션 불가
3. **Cache eviction 등의 고급 기법 적용 불가**: 캐시가 레이어 내부에 숨겨져 있어 정책 변경이 어려움

KVCacheManager는 **모든 attention 레이어의 KV Cache를 호스트 측에서 통합 관리**하는 컨테이너이다. 이 PR은 KVCacheManager 클래스 자체만 도입하며, mha_core와의 실제 연결은 #3904에서 이루어진다.

---

## 어떤 변화가 있는가?

### 1. KVCacheManager 클래스

```cpp
class KVCacheManager {
  struct LayerCache {
    nntrainer::Tensor key_cache;    // (batch, 1, max_seq_len, kv_width)
    nntrainer::Tensor value_cache;  // (batch, 1, max_seq_len, kv_width)
  };
  std::vector<LayerCache> layer_caches_;

  unsigned int cache_pos_ = 0;   // 현재 쓰기 위치
  unsigned int batch_size_;
  unsigned int max_seq_len_;
  unsigned int num_heads_kv_;
  unsigned int head_dim_;
  unsigned int kv_width_;        // num_heads_kv * head_dim
};
```

### 2. 주요 API

| API | 설명 |
|-----|------|
| `allocate(num_layers, batch, max_seq, num_heads_kv, head_dim, dtype, format)` | 모든 레이어의 cache 텐서 할당 |
| `getPosition()` / `setPosition(pos)` / `advance(step)` / `reset()` | 위치 추적 및 제어 |
| `getKeyCache(layer_idx)` / `getValueCache(layer_idx)` | 전체 cache 텐서 접근 |
| `getKeyCacheWriteView(layer, batch, step_size)` | 현재 위치에서의 zero-copy 쓰기 슬라이스 |
| `getValueCacheWriteView(layer, batch, step_size)` | V cache의 zero-copy 쓰기 슬라이스 |
| `getKeyCacheReadView(layer, batch, read_len)` | 처음부터 read_len까지의 zero-copy 읽기 슬라이스 |
| `getValueCacheReadView(layer, batch, read_len)` | V cache 읽기 슬라이스 |
| `save(path)` / `save(path, seq_len)` | cache를 바이너리 파일로 저장 |
| `load(path, seq_len)` | cache를 바이너리 파일에서 복원 |

### 3. Zero-Copy View 메커니즘

**왜 중요한가:** KV Cache 뷰를 복사 없이 전달하면, mha_core가 cache의 올바른 영역을 직접 읽고 쓸 수 있으면서도 메모리는 KVCacheManager가 소유한다.

**WriteView (새로운 K/V를 현재 위치에 기록):**
```cpp
Tensor KVCacheManager::getKeyCacheWriteView(
    unsigned int layer_idx, unsigned int batch, unsigned int step_size) {
  
  auto &cache = layer_caches_[layer_idx].key_cache;
  
  // 오프셋: batch * feature_len + cache_pos_ * kv_width_
  size_t offset = batch * cache_dim.getFeatureLen() + cache_pos_ * kv_width_;
  
  // step_dim = {1, 1, step_size, kv_width_}
  // offset 위치에서부터 shared data tensor로 zero-copy 참조
  return cache.getSharedDataTensor(step_dim, offset, true);
}
```

**ReadView (누적된 시퀀스 전체를 attention 연산용으로 읽기):**
```cpp
Tensor KVCacheManager::getKeyCacheReadView(
    unsigned int layer_idx, unsigned int batch, unsigned int read_len) {
  
  auto &cache = layer_caches_[layer_idx].key_cache;
  
  // 항상 position 0부터 read_len까지
  size_t offset = batch * cache_dim.getFeatureLen();
  return cache.getSharedDataTensor(read_dim, offset, true);
}
```

`getSharedDataTensor()`는 기존 텐서의 메모리를 공유하는 새 텐서를 반환 — **메모리 복사 없음**.

### 4. 저장/로드

```cpp
void KVCacheManager::save(const std::string &path, unsigned int seq_len) {
  for (auto &lc : layer_caches_) {
    // seq_len까지만 저장 (전체 capacity가 아닌 실제 사용분만)
    Tensor k_slice = lc.key_cache.getSharedDataTensor(
        save_dim, 0, true);   // height = seq_len
    Tensor v_slice = lc.value_cache.getSharedDataTensor(
        save_dim, 0, true);
    
    k_slice.save(f);
    v_slice.save(f);
  }
}

void KVCacheManager::load(const std::string &path, unsigned int seq_len) {
  // 각 레이어의 K, V 슬라이스를 읽어 복원
  // cache_pos_ = seq_len으로 설정
}
```

### 5. 테스트 스위트 (17개 테스트)

```
allocate_basic          — 기본 할당
allocate_invalid_params — 유효하지 않은 파라미터로 할당 실패
cache_tensor_dimensions — 캐시 텐서 차원 검증
position_management     — get/set/advance/reset 테스트
position_bounds_check   — 범위를 벗어난 위치 접근 오류
invalid_layer_idx       — 유효하지 않은 레이어 인덱스 접근 오류
write_view_dimensions   — WriteView 차원 검증
read_view_dimensions    — ReadView 차원 검증
write_view_points_to_correct_location — WriteView가 올바른 오프셋을 가리킴
write_and_read_data_consistency       — 쓰기 후 읽기 데이터 일관성
sequential_write_positions            — 순차 쓰기 위치 추적
batch_offset_correct                  — 배치별 오프셋 정확성
multi_layer_independence              — 여러 레이어 간 독립성
save_and_load           — 저장 후 복원
save_load_not_allocated — 할당 전 저장/로드 오류
write_view_overflow     — WriteView 범위 초과 오류
typical_inference_flow  — 전형적인 추론 흐름 (prefill → generate)
```

---

## 기존 코드와의 비교

| 측면 | 내부 cache (이전) | KVCacheManager (이후) |
|------|-------------------|----------------------|
| 소유권 | 각 mha_core 내부 | 호스트(KVCacheManager) |
| 접근 방법 | `model->forEachLayer()` + `context.getTensor()` | `kv_cache.getKeyCacheWriteView(i)` |
| 저장/로드 | 레이어 순회 필요 | `kv_cache.save()` / `kv_cache.load()` |
| 메모리 통합 | 불가능 (분산) | 단일 allocator로 통합 가능 |
| Cache eviction | 불가능 | 설계 훅 제공 (구현은 out of scope) |

---

## 관련 PR

- **#3902**: 이 PR이 의존하는 Tensor API (`Tensor`, `getSharedDataTensor()`)
- **#3904**: KVCacheManager를 mha_core의 5-input external cache 모드와 연결 (PR-B 시리즈의 마지막)