# PR #3837: Base Code - ThreadManager, TorchFXConverter, GraphVisualizer, SafeTensor, Tensor API, Symbolic Graph, 1-bit Quantization Support

- **Author**: jijoongmoon
- **Date**: 2026-04-03
- **Branch**: `claude/analyze-gguf-format-bkBzP` → `main`
- **Size**: +46,862 / -8,699 lines (100 files)
- **Status**: OPEN

---

## 왜 필요한가?

nntrainer에 **대규모 추론 파이프라인**(LLM quantization 기반 추론, 모델 변환, 시각화)과 **코어 아키텍처 현대화**(통합 스레드 관리, 텐서 API 재설계, 안전한 가중치 포맷)를 한 번에 도입하는 "기반 코드" PR이다. 이후 열리는 소형 PR들(#3896, #3898, #3899, #3901, #3902-3904)은 이 PR의 각 컴포넌트를 독립적으로 분리한 것이다.

---

## 어떤 변화가 있는가?

### 1. ThreadManager (통합 스레드 관리)

**이전 코드:**
- `nntr_threads.h`의 `ParallelBatch` (88줄)
- `cache_loader.cpp` (260줄) - 가중치 로딩 전용
- `task_executor.cpp` (193줄) - 비동기 작업
- `bs_thread_pool.h` (2850줄) - 범용 블로킹 스레드 풀

4개의 서로 다른 스레드 구현이 산재해 있어, 각자가 서로 다른 방식으로 코어를 소비하고 CPU 선호도(affinity) 설정도 일관되지 않았다.

**이후 코드:**
`ThreadManager` (`nntrainer/utils/thread_manager.h/.cpp`) 하나로 통합:

```cpp
// Spin-wait 모드 (GGML 스타일, 저지연)
ThreadManager tm;
tm.initialize({.mode = ThreadManager::SPIN_WAIT, .num_threads = 4});
tm.parallel_for(begin, end, [](size_t i) { /* 작업 */ });

// Condvar 모드 (기본 선택, 범용)
tm.initialize({.mode = ThreadManager::CONDVAR});
```

**장점:**
- 두 가지 작동 모드: `SPIN_WAIT` (GGML 스타일 atomic 스핀 배리어, ~0.1μs 디스패치 지연) / `CONDVAR` (표준 condition-variable 기반)
- CPU 선호도(affinity) 제어를 `NNTR_CPU_AFFINITY` 환경변수로 통일
- `#pragma omp` 대신 `tm.parallel_for()`를 사용하여 MoE 레이어의 병렬 연산 통합

### 2. TorchFXConverter

**왜 필요한가:** HuggingFace 모델(Qwen3, Gemma3, LLaMA 등)을 nntrainer 형식으로 변환하는 도구가 필요했다.

**4-Phase 파이프라인:**
1. **Tracer** (`tracer.py`): `torch.fx.Tracer`로 모델 실행 추적, FX 그래프 생성
2. **Decomposer** (`decomposer.py`): 미지원 연산(`rsqrt` → `Tensor::inv_sqrt()` 등) 분해
3. **NodeMapper** (`node_mapper.py`): FX 노드를 nntrainer 레이어 타입으로 매핑
4. **Emitter** (`emitter_cpp/`, `emitter_ini/`): C++ 코드 또는 INI 파일 생성

**지원 모델:**
- Decoder-only CausalLM: Qwen3, LLaMA, Granite (MoE/Hybrid), LFM2, Gemma, Qwen2
- Embedding 모델: Qwen3-Embedding, Gemma-Embedding, KaLM-Embedding, XLM-RoBERTa
- Encoder-decoder: T5Gemma2

### 3. GraphVisualizer (VS Code 확장)

**왜 필요한가:** TorchFXConverter가 만드는 그래프를 시각적으로 확인할 수 있는 도구가 필요했다. Netron과 유사한 VS Code 확장.

**기능:**
- 레이어 트리뷰 (Model Explorer)
- D3.js 기반 인터랙티브 그래프 캔버스
- 컨버터 실행 및 출력 로딩 (`converterRunner.ts`)
- C++ 소스 파일 링크 (노드 클릭 → 해당 소스 코드 파일 열기)

### 4. 1-bit 양자화 (Q1_0)

**왜 필요한가:** PrismML Bonsai 모델 같이 극단적인 압축이 필요한 사례를 위해 1-bit 양자화 지원이 필요했다.

**포맷:** 블록당 128개의 가중치, 1비트씩 패킹 → **18 bytes / 128 weights = 1.125 bpw (bits per weight)**

```
블록 구조 (128 weights):
- 16 bytes: 128개 비트 (가중치 부호)
- 2 bytes: 양자화 스케일 파라미터
```

**커널:**
- **AVX2**: `movemask`로 부호 추출, FMA 기반 내적
- **NEON**: `vbslq_f32`로 비트 선택, `vfmaq_f32`로 누적
- **GGML 인터페이스**: `__ggml_quantize_q1_0`, `__ggml_vec_dot_q1_0_q8_0`, `__ggml_q1_0_GEMM`

### 5. SafeTensors 포맷

**왜 필요한가:** 기존 BIN 포맷은 순차적 오프셋에 의존하여 TorchFXConverter의 FX 실행 순서와 nntrainer의 그래프 순회 순서가 달라지면 **가중치가 잘못 로드**되는 문제가 있었다.

**SafeTensors 레이아웃:**
```
Byte 0        8              8 + header_size
  |           |                    |
  v           v                    v
  [8B 헤더크기][JSON 헤더 (8B 정렬)]  [Raw 텐서 데이터]
```

JSON 헤더에는 각 가중치의 이름, dtype, shape, 오프셋이 포함되어 **이름 기반 조회**가 가능해진다:
```json
{
  "__metadata__": {"format": "nntrainer"},
  "fc1:weight": {"dtype": "F32", "shape": [256, 784], "data_offsets": [0, 802816]}
}
```

`convertBinToSafetensors()` 유틸리티로 BIN → SafeTensors 업그레이드 지원.

### 6. Tensor API / Symbolic Graph

**이전 패턴 (addLayer + string input_layers):**
```cpp
model->addLayer(createLayer("fully_connected",
    {"name=fc1", "unit=128", "activation=relu", "input_layers=x"}));
model->compile();
model->initialize();
```

**이후 패턴 (Symbolic Tensor Graph):**
```cpp
Tensor x({1, 1, 1, 784}, "x");
auto h = createLayer("fully_connected", {"unit=128", "name=fc1"})(x);
auto y = createLayer("fully_connected", {"unit=10", "name=fc2"})(h);
model->compile(x, y);  // 그래프 추출 + compile + initialize + allocate 자동
```

Tensor 상태:
- **Symbolic**: `Tensor({1,1,28,28}, "x")` — 그래프 플레이스홀더, `compile()` 시점에 실체화
- **Eager**: `Tensor::zeros()`, `Tensor::fromData()` — 즉시 데이터 소유

### 7. KleidiAI 통합

Intel의 KleidiAI 라이브러리를 ARM CPU에서의 양자화된 행렬곱(matmul) 가속에 사용. `kleidiai_interface_qai8dxp_qsi4cxp.cpp`, `kleidiai_interface_qsi8d32p_qsi4c32p_omp.cpp` 수정.

### 8. DepthwiseConv1D 레이어

**왜 필요한가:** MobileNet 스타일 아키텍처를 위한 채널별 1D 컨볼루션.

**특징:**
- `groups == in_channels == out_channels` (depthwise separable)
- im2col 버퍼로 가중치를 양자화 친화적 레이아웃으로 변환
- 채널별 dot product (element-wise multiply + sum 대체)

---

## 기존 코드와의 비교

| 영역 | 이전 | 이후 |
|------|------|------|
| 스레드 관리 | 4개 분산 구현 | `ThreadManager` 단일 클래스 |
| 모델 빌드 | `addLayer()` + 문자열 `input_layers` | Symbolic Tensor Graph (`Tensor → LayerHandle → Tensor`) |
| 가중치 포맷 | 순차 오프셋 BIN | JSON 헤더 SafeTensors |
| 모델 변환 | 수동 | TorchFXConverter 자동 파이프라인 |
| 양자화 | Q4_0, Q4_K, Q6_K | + Q1_0 (1.125 bpw) |
| 시각화 | 없음 | VS Code GraphVisualizer 확장 |

---

## 관련 PR

이 PR의 각 컴포넌트는 이후 소형 PR로 분리되어 독립 리뷰됨:
- #3896: Tensor API + Symbolic Graph (분리)
- #3898: SafeTensors 포맷 (분리)
- #3899: ComputeOps 디스패치 테이블 (분리)
- #3901: QNN 통합 (ComputeOps 기반)
- #3902-3904: CausalLM KV Cache 외장화 트랙