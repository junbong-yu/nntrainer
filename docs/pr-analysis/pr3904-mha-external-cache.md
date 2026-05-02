# PR #3904: mha_core 5-Input External KV Cache Mode + Bind from CausalLM

- **Author**: jijoongmoon
- **Date**: 2026-04-27
- **Branch**: `feature/mha-external-cache` → `main`
- **Size**: +12,478 / -2,899 lines (100 files)
- **Status**: OPEN (Wait 4 #3903)
- **Depends on**: #3902 (Symbolic Graph), #3903 (KVCacheManager)

---

## 왜 필요한가?

이 PR은 KV Cache 외장화 트랙(PR-B 시리즈)의 **마지막 단계**다. #3903에서 호스트 측 KVCacheManager를 만들었고, 이 PR에서 실제로 **mha_core가 KVCacheManager의 메모리를 외부 입력으로 받아 사용**하도록 연결한다.

**이전 구조 (내부 cache):**
```
CausalLM::run()
  → model->forwarding()
    → mha_core::forwarding()
      → 내부에서 할당된 cache_key/cache_value 텐서에 K/V 기록
      → 다음 step에서 동일 텐서에서 읽기
```

**이후 구조 (외부 cache):**
```
CausalLM::run()
  → allocateAndBindKVCache()  // KVCacheManager 할당 → placeholder에 바인딩
  → setKVCachePosition(pos)   // 모든 mha_core에 cache_index 전달
  → model->forwarding()
    → mha_core::forwarding()
      → input[3]/input[4]에서 외부 cache 텐서 읽기
      → 외부 cache 텐서에 K/V 기록
      → cache_index 자동 증가
```

---

## 어떤 변화가 있는가?

### 1. mha_core::finalize() — 3/4/5 입력 허용

```cpp
void MHACoreLayer::finalize(InitLayerContext &context) {
  // 3 입력: Q, K, V
  // 4 입력: Q, K, V, mask
  // 5 입력: Q, K, V, cache_key, cache_value  ← NEW
  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 5, ...);

  use_external_cache = (context.getNumInputs() >= 5);

  if (!use_external_cache) {
    // 외부 cache 모드가 아닐 때만 내부 cache 텐서 할당
    tensor_idx[cache_key] = context.requestTensor(cache_key_dim, ...);
    tensor_idx[cache_value] = context.requestTensor(cache_value_dim, ...);
  }
  // 외부 cache 모드: requestTensor 호출 안 함 → 호스트가 외부에서 제공
}
```

### 2. 새로운 Public API

```cpp
// mha_core.h
void setCacheIndex(unsigned int idx);   // 호스트가 현재 쓰기 위치 설정
unsigned int getCacheIndex() const;     // 현재 위치 조회
```

### 3. forwarding() 구현 (외부 cache 모드)

```cpp
void MHACoreLayer::forwarding(RunLayerContext &context, bool training) {
  if (!use_external_cache) return;  // 기존 경로는 그대로

  Tensor &query      = context.getInput(0);  // Q
  Tensor &key        = context.getInput(1);  // K
  Tensor &value      = context.getInput(2);  // V
  Tensor &cache_key  = context.getInput(3);  // 외부 K cache
  Tensor &cache_value = context.getInput(4); // 외부 V cache

  unsigned int step_size = query.height();
  unsigned int from = cache_index;
  unsigned int to = cache_index + step_size;

  // 각 배치별로:
  // 1. Q, K, V + output의 step-size 슬라이스 추출
  // 2. 외부 cache 텐서 + [from, to) 범위로 one_batch_incremental_forwarding() 호출
  //    → K, V를 cache의 [from, to) 영역에 기록
  //    → cache[0, from)의 내용으로 attention 수행
  //    → FP32, FP16, Android 경로 모두 동일한 one_batch_incremental_forwarding() 사용

  cache_index += step_size;  // 다음 step을 위해 자동 증가
}
```

### 4. incremental_forwarding() 라우팅

```cpp
void MHACoreLayer::incremental_forwarding(RunLayerContext &context,
                                          unsigned int _from, unsigned int _to,
                                          bool training) {
  // 외부 cache 모드: _from을 cache_index로 설정하고 forwarding()에 위임
  if (use_external_cache) {
    cache_index = _from;
    forwarding(context, training);
    return;
  }
  
  // 내부 cache 모드: 기존 구현 그대로 유지
  // ...
}
```

**중요:** 호출자(NeuralNetwork::incremental_inference)는 변경 없음 — `incremental_forwarding(_from, _to)`를 그대로 호출하고, 내부에서 외부/내부 분기.

### 5. Graph Wiring: createKVCachePlaceholders()

```cpp
// transformer.cpp — Transformer 기본 클래스
std::pair<Tensor, Tensor>
Transformer::createKVCachePlaceholders(const int layer_id, int n_heads) {
  
  const unsigned int max_timestep = INIT_SEQ_LEN + NUM_TO_GENERATE;
  const unsigned int kv_width = HEAD_DIM * n_heads / GQA_SIZE;

  TensorDim cache_dim({BATCH_SIZE, 1, max_timestep, kv_width},
                       {TensorDim::Format::NCHW, TensorDim::DataType::FP16});

  // 이름 있는 placeholder 텐서 — 모델 전체에서 고유한 이름
  Tensor cache_k(cache_dim, "cache_k_l" + std::to_string(layer_id));
  Tensor cache_v(cache_dim, "cache_v_l" + std::to_string(layer_id));
  
  return {cache_k, cache_v};
}
```

### 6. createAttention에서 Placeholder 연결

모든 Transformer 파생 클래스(Qwen2, Qwen3, Gemma3, GPT-OSS 등)의 `createAttention`이 placeholder를 요청하고 5-input mha_core에 전달:

```cpp
Tensor Transformer::createAttention(int layer_id, int seq_len, int n_heads,
                                     int head_dim, Tensor query, Tensor key,
                                     Tensor value) {
  // 각 레이어별 KV cache placeholder 생성
  auto [cache_k, cache_v] = createKVCachePlaceholders(layer_id, n_heads);
  
  LayerHandle mha(createLayer("mha_core", {...}));
  
  // 5개 입력: Q, K, V, cache_K, cache_V
  Tensor attn = mha({q, k, v, cache_k, cache_v});
  
  return wo(attn);
}
```

MoE 변형들(qwen3_moe, qwen3_slim_moe, qwen3_cached_slim_moe)은 Qwen3Transformer의 `createAttention`을 상속받아 **자동으로 적용**된다.

### 7. CausalLM Lifecycle

#### allocateAndBindKVCache()
```cpp
void CausalLM::allocateAndBindKVCache() {
  if (kv_cache.isAllocated()) return;  // idempotent

  // 1. KVCacheManager에 모든 레이어의 cache 메모리 할당
  kv_cache.allocate(NUM_LAYERS, BATCH_SIZE, max_timestep,
                    NUM_KEY_VALUE_HEADS, HEAD_DIM, cache_dtype);

  // 2. 각 레이어의 placeholder를 실제 KVCacheManager 버퍼에 바인딩
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto &kc = kv_cache.getKeyCache(i);
    auto &vc = kv_cache.getValueCache(i);
    
    auto *kp = model->getTensor("cache_k_l" + std::to_string(i));
    auto *vp = model->getTensor("cache_v_l" + std::to_string(i));
    
    kp->setData(kc.getMemoryData(), kc.getOffset(), false);  // 포인터 패치
    vp->setData(vc.getMemoryData(), vc.getOffset(), false);
  }
}
```

**바인딩 흐름:**
1. `createKVCachePlaceholders()`가 그래프에 `"cache_k_l{i}"`라는 이름의 symbolic 텐서를 만듦
2. `allocateAndBindKVCache()`가 `model->getTensor("cache_k_l{i}")`로 placeholder를 찾음
3. `setData()`로 placeholder의 메모리 포인터를 KVCacheManager 버퍼로 변경
4. 이후 mha_core가 `input[3]`으로 읽을 때 실제로는 KVCacheManager의 메모리

#### setKVCachePosition()
```cpp
void CausalLM::setKVCachePosition(unsigned int pos) {
  kv_cache.setPosition(pos);  // KVCacheManager 위치 설정
  
  // 모든 mha_core 레이어에도 cache_index 전파
  model->forEachLayer([pos](Layer *layer) {
    if (auto *mha = dynamic_cast<MHACoreLayer*>(layer)) {
      mha->setCacheIndex(pos);
    }
  });
}
```

#### run() — 추론 루프
```cpp
void CausalLM::run() {
  allocateAndBindKVCache();  // 첫 호출 시에만 할당, 이후 idempotent
  setKVCachePosition(0);     // 모든 cache_index = 0
  
  // prefill (첫 토큰)
  model->forwarding(...);
  
  // generate (후속 토큰들)
  for (int step = 1; step < NUM_TO_GENERATE; step++) {
    setKVCachePosition(step);  // 모든 mha_core에 새 position 전달
    model->forwarding(...);    // mha_core가 외부 cache에 K/V 기록, cache_index 자동 증가
  }
}
```

### 8. save_kvcache / load_kvcache 재작성

**이전 (내부 cache 접근 — forEachLayer 순회):**
```cpp
void CausalLM::save_kvcache(std::string path, int to_) {
  auto f = ...;
  model->forEachLayer([](Layer *layer, void *arg) {
    // 각 mha_core의 context.getTensor(0), getTensor(1)로 내부 텐서 접근
    // 복잡한 순회 + 텐서 추출 로직
  });
}

void CausalLM::load_kvcache(std::string path, int to_) {
  model->allocate(INFERENCE);  // context.getTensor() 호출 전 필요
  // 유사한 forEachLayer 순회
}
```

**이후 (KVCacheManager에 위임):**
```cpp
void CausalLM::save_kvcache(std::string path, int to_) {
  kv_cache.save(path, to_);  // 1줄
}

void CausalLM::load_kvcache(std::string path, int to_) {
  if (!kv_cache.isAllocated()) allocateAndBindKVCache();
  kv_cache.load(path, to_);       // KVCacheManager가 직접 파일 I/O
  setKVCachePosition(to_);        // 복원된 위치로 동기화
}
```

---

## 기존 코드와의 비교

| 측면 | 내부 cache (이전) | 외부 cache (이후) |
|------|-------------------|-------------------|
| Cache 소유권 | 각 mha_core 내부 | KVCacheManager |
| mha_core 입력 수 | 3 (Q, K, V) | 5 (Q, K, V, cache_K, cache_V) |
| Cache 할당 | mha_core::finalize()에서 requestTensor | skip → 호스트가 external로 제공 |
| Cache 접근 | context.getTensor(cache_key_idx) | context.getInput(3/4) |
| 위치 추적 | mha_core 내부 cache_index | 호스트가 setCacheIndex()로 제어 |
| 저장/로드 | model->forEachLayer() 순회 | kv_cache.save() / kv_cache.load() |
| Cache eviction | 불가능 | KVCacheManager에 정책 추가 가능 |

---

## 세 PR(#3902, #3903, #3904)의 의존 관계

```
PR #3902 (기반)
├── Transformer/CausalLM을 Tensor API로 마이그레이션
├── createKVCachePlaceholders() API 생성
├── constructModel() → {input, output} 반환
└── compile(x, y, INFERENCE) 단일 호출

        ↓ (의존)

PR #3903 (스토리지)
├── KVCacheManager 클래스
├── zero-copy WriteView/ReadView
├── save/load 바이너리 직렬화
└── 독립 실행 가능 — 아직 mha_core와 연결 안 됨

        ↓ (의존)

PR #3904 (런타임 연결)
├── mha_core::finalize() 5-input 모드
├── forwarding()에서 외부 cache 읽기/쓰기
├── allocateAndBindKVCache()로 placeholder ↔ KVCacheManager 바인딩
├── setKVCachePosition()로 cache_index 전파
└── save_kvcache/load_kvcache 재작성
```

---

## 향후 작업 (Out of Scope)

- Cache eviction 정책 (sliding window, LRU 등)
- Paged attention (vLLM 스타일 블록 단위 캐시 관리)
- Cache compression (FP16 → INT8 등)
- QNN/HMX NPU에서의 external cache 최적화