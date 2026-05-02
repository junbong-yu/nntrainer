# PR #3901: QNN (Qualcomm AI Engine Direct) Integration

- **Author**: jijoongmoon
- **Date**: 2026-04-27
- **Branch**: `feature/qnn-integration` → `main`
- **Size**: +10,760 / -2,234 lines (93 files)
- **Status**: OPEN (Wait 4 #3899)
- **Depends on**: #3899 (ComputeOps 디스패치 테이블)

---

## 왜 필요한가?

Qualcomm Snapdragon 디바이스의 Hexagon NPU는 양자화된 뉴럴 네트워크 추론에서 CPU 대비 **10~100배의 성능-와트 비율**을 제공한다. nntrainer가 모바일 추론 프레임워크로서 이 하드웨어를 활용할 수 있어야 한다.

QNN(Qualcomm AI Engine Direct) SDK는 Hexagon NPU로 모델을 오프로드하는 그래프-컴파일 백엔드로, `QNNLinear`, `QNNGraph` 같은 레이어를 통해 QNN 가속을 nntrainer에 통합한다.

---

## 어떤 변화가 있는가?

### 1. QNNContext: Context + Singleton 패턴

```cpp
class QNNContext : public Context, public Singleton<QNNContext> {
  QNNContext() : Context(std::make_shared<QNNBackendVar>()) {}

  void initialize() noexcept {
    init_backend();                              // CPU 백엔드 초기화
    getContextData()->setComputeOps(g_compute_ops); // CPU 폴백 ops 등록
    init();                                      // QNN SDK 초기화
    setMemAllocator(std::make_shared<QNNRpcManager>()); // DSP 공유 메모리
  }
};
```

`AppContext`, `ClContext`와 동일한 Singleton 패턴을 사용하여 프로세스 전역 단일 인스턴스 보장.

### 2. CPU 폴백 배선

QNN 그래프에 포함되지 않은 텐서 연산(예: 토크나이저, 전처리)은 CPU에서 실행된다:

```cpp
// QNNContext::initialize()
getContextData()->setComputeOps(g_compute_ops);
```

`g_compute_ops`는 #3899에서 도입된 ComputeOps 런타임 디스패치 테이블. QNN 백엔드가 지원하지 않는 연산은 자동으로 CPU로 폴백.

### 3. 플러그인 아키텍처 — `libqnn_context.so`

QNN 코드는 **별도 공유 라이브러리**(`libqnn_context.so`)로 컴파일되어, 메인 `libnntrainer`가 Qualcomm SDK에 직접 링크되지 않는다:

```cpp
// qnn_context.cpp
extern "C" {
nntrainer::ContextPluggable ml_train_context_pluggable{
    create_qnn_context,   // factory: () → Context*
    destory_qnn_context   // destructor
};
}
```

### 4. 엔진 등록

```cpp
// engine.cpp — Engine::add_default_object()
#if defined(ENABLE_NPU) && ENABLE_NPU == 1
  try {
    registerContext("libqnn_context.so", "");
    // 빈 config → dlopen("libqnn_context.so") → ml_train_context_pluggable 검색
  } catch (std::exception &e) {
    ml_logw("QNN context plugin not available: %s", e.what());
    // QNN SDK가 없어도 앱이 죽지 않음
  }
#endif
```

**장점:**
- `enable-npu=true`일 때만 QNN 코드가 빌드됨
- QNN SDK가 없어도 로그 경고만 출력하고 정상 실행
- 46K LOC의 Qualcomm 기밀 헤더가 메인 빌드에 포함되지 않음

### 5. 주요 컴포넌트

| 컴포넌트 | 파일 | 역할 |
|----------|------|------|
| `QNNContext` | `qnn_context.h/.cpp` | QNN 런타임 컨텍스트 (Singleton + Context) |
| `QNNBackendVar` | `qnn/jni/qnn_context_var.h` | ContextData 서브클래스 (QNN 상태 보유) |
| `QNNRpcManager` | `qnn/jni/qnn_rpc_manager.cpp` | MemAllocator: `rpcmem_alloc/free`로 DSP 공유 메모리 관리 |
| `IOTensorWrapper` | `qnn/jni/iotensor_wrapper.hpp` | QNN 텐서 등록 및 입출력 처리 |
| `QNNLinear` | `qnn/jni/qnn/op/QNNLinear.cpp` | QNN fully-connected 레이어 |
| `QNNGraph` | `qnn/jni/qnn/op/QNNGraph.cpp` | QNN HTP 그래프 컴파일 및 실행 |

### 6. 코드 클린업 (3건)

이 PR은 원본 pr/3826의 코드를 가져오면서 3가지 중요 수정을 포함:

#### A. Iterator Invalidation 버그 수정
```cpp
// 이전: range-for 내에서 map.erase() → UB
~QNNRpcManager() {
  for (auto &mem : ptrToFdAndMemHandleMap_) {
    rpcmem_free(mem.first);
    ptrToFdAndMemHandleMap_.erase(mem.first); // 반복자 무효화!
  }
}

// 이후: 루프 밖에서 clear()
~QNNRpcManager() {
  for (auto &mem : ptrToFdAndMemHandleMap_) {
    rpcmem_free(mem.first);
  }
  ptrToFdAndMemHandleMap_.clear(); // 소멸자가 어차피 정리하므로 제거도 가능
}
```

#### B. std::cout → ml_logd (5건)
```cpp
// 이전
std::cout << "QNNContext::init called" << std::endl;

// 이후
ml_logd("QNNContext::init called");
```

#### C. exit(1) → return -1 (3건)
```cpp
// 이전: 라이브러리 함수가 exit() 호출 → 호스트 프로세스 강제 종료
if (!log::initializeLogging()) {
  ml_loge("ERROR: Unable to initialize logging!");
  exit(1);  // 복구 경로 없음
}

// 이후: 오류 코드 반환
if (!log::initializeLogging()) {
  ml_loge("ERROR: Unable to initialize logging!");
  return -1;  // 호출자가 오류 처리 가능
}
```

### 7. 빌드 시스템

```meson
# meson_options.txt
option('enable-npu', type: 'boolean', value: false,
       description: 'Enable QNN (Qualcomm AI Engine Direct) NPU support')

# nntrainer/meson.build
if get_option('enable-npu')
  # QNN subdir는 nntrainer_dep 선언 이후에 포함되어야 함
  # → libqnn_context.so가 libnntrainer에 링크되도록
  subdir('qnn')
endif
```

---

## 기존 코드와의 비교

| 측면 | AppContext만 존재 (이전) | QNN 통합 (이후) |
|------|--------------------------|-----------------|
| 백엔드 | CPU만 (ARM NEON/x86 AVX) | CPU + QNN NPU |
| 메모리 | 표준 malloc/free | rpcmem_alloc (DSP 공유 영역) |
| 링크 의존성 | 없음 | QNN SDK (enable-npu=true일 때만) |
| 오류 처리 | 없음 | exit(1) → return -1, std::cout → ml_logd |
| 빌드 포함 | 항상 | enable-npu=true일 때만 |

---

## 관련 PR

- #3899: ComputeOps 디스패치 테이블 (QNNContext가 CPU 폴백으로 사용)
- #3896: Tensor API (ContextData 패턴)
- #3902-3904: CausalLM KV Cache (CausalLM이 QNNContext에서도 동작 가능)