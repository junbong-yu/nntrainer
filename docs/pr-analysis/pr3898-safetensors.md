# PR #3898: SafeTensors Format Support for NeuralNetwork Save/Load

- **Author**: jijoongmoon
- **Date**: 2026-04-24
- **Branch**: `feature/safetensors` → `main`
- **Size**: +1,227 / -14 lines (13 files)
- **Status**: OPEN

---

## 왜 필요한가?

nntrainer의 기존 가중치 파일 포맷(BIN)은 **순차적 오프셋** 방식으로, TorchFXConverter가 변환한 가중치 파일의 바이트 순서와 NeuralNetwork가 그 파일을 읽는 순서가 일치해야만 정상적으로 로딩된다. 그러나 TorchFXConverter는 PyTorch의 FX 실행 순서로 파일을 작성하고, NeuralNetwork는 토폴로지컬 정렬(topo-sort) 순서로 읽기 때문에 **가중치가 잘못된 레이어에 로드**되는 버그가 발생했다.

SafeTensors 포맷은 JSON 헤더에 각 가중치의 **이름, dtype, shape, 바이트 오프셋**을 저장하여, 로더가 헤더를 읽고 **이름 기반으로 오프셋을 조회**할 수 있게 한다. 파일 작성 순서와 읽기 순서가 달라도 정확히 일치하는 가중치를 찾아 로드한다.

---

## 어떤 변화가 있는가?

### 1. SafeTensors 파일 포맷

```
Byte 0                    8              8 + header_size
  |                       |                    |
  v                       v                    v
  +----------+---------------------------+-----------------------+
  | header   |       JSON 헤더            |     Raw 텐서 데이터    |
  | size     |    (8바이트 정렬,         |                       |
  | (8B LE)  |     공백으로 패딩)         |                       |
  +----------+---------------------------+-----------------------+
```

**JSON 헤더 예시:**
```json
{
  "__metadata__": {"format": "nntrainer"},
  "fc1:weight": {
    "dtype": "F32",
    "shape": [256, 784],
    "data_offsets": [0, 802816]
  },
  "fc1:bias": {
    "dtype": "F32",
    "shape": [256],
    "data_offsets": [802816, 803840]
  }
}
```

- `shape`에서 leading 1은 제거 (`[1,1,512,768]` → `[512,768]`)
- `data_offsets`는 `[start, end)` 로 raw 데이터 섹션 내 바이트 범위

### 2. Dtype 매핑

| nntrainer 타입 | SafeTensors dtype |
|----------------|-------------------|
| FP32 | `"F32"` |
| FP16 | `"F16"` |
| QINT4 | `"I4"` |
| QINT8 | `"I8"` |
| QINT16 | `"I16"` |
| UINT4 | `"U4"` |
| UINT8 | `"U8"` |
| UINT16 | `"U16"` |
| UINT32 | `"U32"` |

### 3. `NeuralNetwork::save()` — SafeTensors vs BIN

**이전 (BIN, 순차 오프셋):**
```cpp
// 헤더 없음. 토폴로지컬 순서로 바이트를 순차 기록.
// 작성자와 읽는 자가 반드시 같은 순서를 따라야 함.
model_file.write(weight_data, weight_size);
model_file.write(next_weight_data, next_weight_size);
// ...
```

**이후 (SafeTensors):**
```cpp
// 1. 모든 가중치의 entry(name, dtype, shape, offsets)를 수집
std::vector<TensorEntry> entries;
for (auto &node : model_graph) {
    for (auto &w : node->getWeights()) {
        entries.push_back({w.getName(), dtypeToString(w.getDataType()),
                           getShape(w), offset, offset + sz});
        offset += sz;
    }
}

// 2. JSON 헤더 빌드 + 8B 크기 접두사
auto header = nntrainer::safetensors::buildHeader(entries);
uint64_t header_size = header.size();
model_file.write(&header_size, 8);
model_file.write(header.data(), header.size());

// 3. Raw 텐서 데이터 기록
for (auto &node : model_graph) {
    node->save(model_file, false, exec_mode);
}
```

### 4. `NeuralNetwork::load()` — 이름 기반 오프셋 조회

**이전 (BIN):**
```cpp
// 순차 오프셋 누적 — 순서에 의존
start_from = 0;
for (auto &node : model_graph) {
    for (auto &w : node->getWeights()) {
        w->setFileOffset(start_from);
        start_from += w->getMemoryBytes();
    }
}
```

**이후 (SafeTensors):**
```cpp
// 1. 첫 8바이트에서 헤더 크기 읽기
uint64_t header_size;
probe.read(&header_size, 8);

// 2. JSON 헤더 파싱 → name → {offset, size} 맵 생성
auto name_offset_map = nntrainer::safetensors::parseHeader(header_json);
auto data_start = 8 + header_size;

// 3. 각 가중치를 이름으로 조회 — 순서 무관!
for (auto &node : model_graph) {
    for (auto &w : node->getWeights()) {
        auto it = name_offset_map.find(w->getName());
        if (it != name_offset_map.end()) {
            w->setFileOffset(data_start + it->second.first);
        } else {
            // 이름이 없으면 순차 오프셋으로 폴백 (경고 로깅)
            ml_logw("Weight '%s' not found in safetensors header");
            w->setFileOffset(start_from);
        }
    }
}
```

### 5. Multi-threaded mmap 로딩 (INFERENCE 모드)

**왜 필요한가:** LLM 추론에서는 수백 MB~수 GB의 가중치를 로드해야 한다. 순차 로딩은 느리고 메모리를 낭비한다.

```cpp
// INFERENCE 모드일 때만 mmap + 멀티스레드 로딩
std::vector<std::thread> threads;
for (auto &node : model_graph) {
    threads.emplace_back([&, node]() {
        int fd = ::open(file_path.c_str(), O_RDONLY);
        void *mmap_ptr = ::mmap(nullptr, file_size, PROT_READ,
                                 MAP_PRIVATE, fd, 0);
        ::close(fd);  // mmap 후 fd 불필요
        ::posix_madvise(mmap_ptr, file_size, POSIX_MADV_RANDOM);
        
        node->read(static_cast<char*>(mmap_ptr), ...);
        
        ::posix_madvise(mmap_ptr, file_size, POSIX_MADV_DONTNEED);
        ::munmap(mmap_ptr, file_size);
    });
}
for (auto &t : threads) t.join();
```

- 레이어당 하나의 스레드, 각자가 같은 파일을 mmap (MAP_PRIVATE, 커널이 페이지 공유)
- `POSIX_MADV_RANDOM` → 분산 접근 패턴 힌트
- `POSIX_MADV_DONTNEED` → 읽은 페이지 즉시 해제 (메모리 절약)
- TRAINING 모드에서는 기존처럼 순차 ifstream 로딩 사용

### 6. `convertBinToSafetensors()` 유틸리티

```cpp
void NeuralNetwork::convertBinToSafetensors(
    const std::string &bin_path, const std::string &st_path) {
  load(bin_path, MODEL_FORMAT_BIN);        // BIN → 모델에 로드
  save(st_path, MODEL_FORMAT_SAFETENSORS); // 모델 → SafeTensors로 저장
}
```

BIN 포맷에서 SafeTensors로의 **in-place 업그레이드**를 지원.

### 7. 자동 포맷 감지

CausalLM의 `load_weight()`는 파일 확장자로 포맷을 자동 선택:
```cpp
if (weight_path.ends_with(".safetensors")) {
    format = MODEL_FORMAT_SAFETENSORS;
} else {
    format = MODEL_FORMAT_BIN;
}
```

---

## 기존 코드와의 비교

| 측면 | BIN v0 (이전) | SafeTensors (이후) |
|------|---------------|-------------------|
| 헤더 | 없음 | 8B 크기 + JSON |
| 가중치 조회 | 순차 누적 오프셋 | 이름 기반 JSON 조회 |
| 순서 의존성 | 엄격한 토폴로지컬 순서 필요 | 순서 무관 |
| 병렬 로딩 | 레이어별 스레드 (ifstream) | 레이어별 스레드 (mmap, zero-copy) |
| 메타데이터 | 없음 | dtype, shape, offset |
| 오류 진단 | 정확히 알 수 없음 | 이름 불일치 → 경고 로그 |

---

## 관련 PR

- #3837: SafeTensors 최초 도입 (이 PR은 그 컴포넌트를 분리)
- #3896: Tensor API (SafeTensors 헤더의 `dtypeToString`이 TensorDim::DataType 사용)