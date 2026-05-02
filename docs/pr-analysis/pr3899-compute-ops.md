# PR #3899: ComputeOps Function-Pointer Dispatch Table (CPU Backends)

- **Author**: jijoongmoon
- **Date**: 2026-04-25
- **Branch**: `feature/compute-ops-core` → `main`
- **Size**: +7,657 / -2,233 lines (74 files)
- **Status**: OPEN (Wait 4 #3896)
- **Depends on**: #3896 (Tensor API 기반 코드)

---

## 왜 필요한가?

기존 nntrainer의 텐서 연산 디스패치는 `#ifdef` 기반으로 **컴파일 타임에만 결정**되었다:

```cpp
// cpu_backend.h (기존)
#if defined(__aarch64__) || defined(__ARM_ARCH_7A__)
  #include <arm_compute_backend.h>   // ARM NEON 구현
#elif defined(__x86_64__)
  #include <x86_compute_backend.h>   // AVX2 구현
#else
  #include <fallback.h>              // 순수 C++ 폴백
#endif
```

**문제점:**
1. 백엔드 추가 때마다 `cpu_backend.h` 수정 필요 (QNN, CUDA, OpenCL 등)
2. 런타임에 백엔드 전환 불가능
3. 서로 다른 백엔드의 텐서가 섞여도 감지 불가 → 잘못된 메모리 접근 (silent corruption)

PR #3899는 이 `#ifdef` 디스패치를 **런타임 가상 함수 테이블**로 대체하는 인프라를 구축한다.

---

## 어떤 변화가 있는가?

### 1. `nntrainer::ComputeOps` 추상 인터페이스

`compute_ops.h`에 약 **60개의 가상 메서드**를 가진 추상 클래스:

```cpp
class ComputeOps {
public:
  // FP32 BLAS
  virtual void sgemm_fp32(...) = 0;
  virtual void sgemv_fp32(...) = 0;
  virtual float sdot_fp32(...) = 0;

  // FP32 Element-wise
  virtual void ele_mul_fp32(...) = 0;
  virtual void ele_add_fp32(...) = 0;

  // FP32 Activation
  virtual void swiglu_fp32(...) = 0;
  virtual void softmax_fp32(...) = 0;

  // Quantized GEMM
  virtual void gemm_q4_0_fp32(...) = 0;
  virtual void gemm_q6_K_fp32(...) = 0;

  // FP16 variants
  virtual void sgemm_fp16(...) = 0;
  // ... 등 약 60개

  // Accelerator-only (GPU/NPU 전용)
  virtual bool supports_gemm_q4_0_batch() const { return false; }
  virtual void gemm_q4_0_batch_fp32(...) { throw; }
  // ...
};
```

**왜 struct가 아니라 class인가:** 가상 함수를 통해 각 백엔드가 `cl_command_queue`, `npu_session` 같은 자신만의 상태를 멤버 변수로 가질 수 있다. 일반 함수 포인터 struct로는 `this`를 전달할 수 없어서 불가능하다.

### 2. Global `g_compute_ops` + Thread-Safe 초기화

```cpp
// compute_ops.cpp
ComputeOps *g_compute_ops = nullptr;
static std::once_flag g_compute_ops_init_flag;

void ensureComputeOps() {
  std::call_once(g_compute_ops_init_flag, []() {
    init_backend();  // ARM/x86/fallback 중 하나의 init_backend() 호출
  });
}

ComputeOps *getComputeOps() {
  if (__builtin_expect(g_compute_ops == nullptr, 0))  // fast path
    ensureComputeOps();
  return g_compute_ops;
}
```

`std::call_once`로 스레드 안전한 단일 초기화 보장.

### 3. CPU Ops Table

`cpu_ops_table.cpp`가 **모든 CPU 아키텍처를 위한 단일 CpuComputeOps**를 제공:

```cpp
class CpuComputeOps : public ComputeOps {
  void sgemm_fp32(...) override { nntrainer::sgemm(...); }
  void ele_mul_fp32(...) override { nntrainer::ele_mul(...); }
  // ... 모든 60개 메서드를 nntrainer:: 함수로 포워딩
};

ComputeOps *get_cpu_ops() {
  static CpuComputeOps instance;
  return &instance;
}
```

각 아키텍처의 `compute_backend.cpp`가 `init_backend()`에서 `g_compute_ops = get_cpu_ops()`를 호출한다. 링크 타임에 올바른 `nntrainer::sgemm` (ARM NEON / x86 AVX / fallback) 심볼이 해석된다.

### 4. OpenCL Ops Table (`cl_compute_ops.cpp`)

```cpp
class ClComputeOps : public ComputeOps {
  // 가속기 전용 연산만 오버라이드
  bool supports_gemm_q4_0_batch() const override { return true; }
  void gemm_q4_0_batch_fp32(...) override {
    nntrainer::gemm_q4_0_async_cl(...);  // OpenCL 커널 호출
  }
  // 나머지 연산은 기본 구현 → 예외 throw
};
```

### 5. ContextData 분리와 `as<T>()` 템플릿

**문제:** `context.h`가 `layer_devel.h`를 포함하고, `layer_devel.h`가 `context.h`를 포함하는 순환 의존성.

**해결책:** `ContextData` 클래스를 `context_data.h`로 분리하여 순환 의존성 해소.

```cpp
// context_data.h
class ContextData {
  ComputeOps *compute_ops_ = nullptr;
  std::shared_ptr<MemAllocator> mem_allocator_;

public:
  ComputeOps *getComputeOps() { return compute_ops_; }
  void setComputeOps(ComputeOps *ops) { compute_ops_ = ops; }

  // 타입 안전한 벤더 다운캐스트
  template <typename T> T *as() { return dynamic_cast<T *>(this); }
};
```

**사용 예시 (QNN):**
```cpp
auto *qnn_ctx = context.getContextData()->as<QNNBackendVar>();
// QNN 전용 API 사용 가능
```

### 6. AppContext 연동

```cpp
// app_context.cpp
void AppContext::initialize() {
  ensureComputeOps();  // g_compute_ops 보장
  if (auto cd = getContextData(); cd && g_compute_ops) {
    cd->setComputeOps(g_compute_ops);  // ContextData에 전역 ops 등록
  }
}
```

### 7. 텐서 연산 호출 경로 (미래)

이 PR은 인프라만 구축. **실제 텐서 연산을 테이블로 라우팅하는 부분은 후속 PR**에서 처리:

```
미래 호출 경로:
  tensor.dot(other)
    → tensor.getContextData()
       → cd->getComputeOps()->sgemm_fp32()  // 벤더 전용 구현
       → OR: getComputeOps()->sgemm_fp32()  // 전역 CPU 폴백
    
    → 서로 다른 ContextData → throw (크로스-벤더 보호)
```

---

## 기존 코드와의 비교

| 측면 | `#ifdef` 디스패치 (이전) | ComputeOps (이후) |
|------|-------------------------|-------------------|
| 디스패치 시점 | 컴파일 타임 | 런타임 (가상 함수) |
| 백엔드 추가 | `cpu_backend.h`에 `#ifdef` 브랜치 추가 | `ComputeOps` 상속 클래스 추가 (플러그인 가능) |
| 크로스-벤더 검사 | 불가능 | ContextData 불일치 시 예외 발생 |
| 벤더 상태 관리 | 없음 | 멤버 변수로 벤더별 상태 보유 가능 |
| 핫스왑 | 불가능 | 런타임에 `setComputeOps()`로 전환 |

---

## 관련 PR

- #3896: Tensor API (ComputeOps가 텐서에 부착되는 ContextData 사용)
- #3901: QNN 통합 (QNNContext가 자체 ComputeOps를 ContextData에 등록)
- 후속 PR: `float_tensor.cpp` / `tensor.cpp`에서 직접 호출을 ComputeOps 가상 호출로 전환