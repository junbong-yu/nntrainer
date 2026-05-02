# PR #3896: ml::train::Tensor API with Pimpl + Symbolic Graph Compile

- **Author**: jijoongmoon
- **Date**: 2026-04-24
- **Branch**: `feature/tensor-api` → `main`
- **Size**: +4,624 / -1,645 lines (24 files)
- **Status**: OPEN (Wait 4 #3896)
- **Depends on**: #3837 (Tensor API 기반 코드)

---

## 왜 필요한가?

nntrainer의 모델 빌드 방식을 **문자열 기반**에서 **타입 안전한 Symbolic Tensor Graph**로 전환한다. 기존에는 `addLayer()`로 레이어를 추가하고 `"input_layers=layer1,layer2"` 같은 문자열로 연결을 명시했는데, 이 방식은 오타에 취약하고 리팩토링이 어려우며 IDE 지원을 받을 수 없었다.

새로운 API는 `Tensor → LayerHandle → Tensor` 형태의 함수형 체이닝을 통해 그래프를 구축한다. 이는 PyTorch나 JAX의 스타일과 유사하며, 컴파일 타임에 연결 오류를 잡을 수 있다.

---

## 어떤 변화가 있는가?

### 1. Tensor 클래스 재설계 (Pimpl 패턴)

**이전 코드:**
```cpp
// 이전: Var_Grad를 직접 상속, 내부 상태 노출
class Tensor : public nntrainer::Var_Grad {
public:
  Tensor() : nntrainer::Var_Grad() {}
  // var, grad 등이 public으로 노출됨
};
```

**이후 코드:**
```cpp
// 이후: Pimpl로 내부 구현 완전 은닉
class Tensor {
  struct Impl;                      // 전방 선언 (비공개)
  std::unique_ptr<Impl> impl_;     // 불투명 포인터
};
```

`Impl` 내부 구조 (`tensor_api_impl.h`, 설치되지 않는 비공개 헤더):
- `TensorDim dim` — 차원 정보
- `std::string name` — 텐서 이름
- `std::shared_ptr<SymbolicGraphNode> graph_edge` — 그래프 엣지 (symbolic 모드)
- `nntrainer::Tensor *bound_tensor` — 컴파일 후 바인딩된 실제 텐서
- `std::vector<std::function<...>> call_chain` — 지연 연산 큐

### 2. Symbolic vs Eager 텐서

| 상태 | 생성 방법 | `isMaterialized()` | 용도 |
|------|-----------|-------------------|------|
| **Symbolic** | `Tensor({1,1,28,28}, "x")` | `false` → `compile()` 후 `true` | 그래프 플레이스홀더 |
| **Eager** | `Tensor::zeros({...})`, `Tensor::fromData(ptr, dim)` | 즉시 `true` | 직접 데이터 연산 |

### 3. 새로운 `Model::compile(Tensor, Tensor)` API

**이전 패턴 (문자열 기반):**
```cpp
// mnist 예제 (이전)
model->addLayer(createLayer("input", {"name=inputlayer", ...}));
model->addLayer(createLayer("conv2d", {"name=conv2d_c1_layer", "input_layers=inputlayer", ...}));
model->addLayer(createLayer("pooling2d", {"name=pooling2d_p1", "input_layers=conv2d_c1_layer", ...}));
model->addLayer(createLayer("fully_connected", {"name=outputlayer", "input_layers=flatten", ...}));
model->compile();
model->initialize();
model->allocate();
```

**이후 패턴 (Symbolic Tensor Graph):**
```cpp
// mnist 예제 (이후)
auto x = Tensor({1, 1, 28, 28}, "inputlayer");

LayerHandle conv1(createLayer("conv2d", {"name=conv2d_c1_layer", ...}));
LayerHandle pool1(createLayer("pooling2d", {"name=pooling2d_p1", ...}));
LayerHandle fc(createLayer("fully_connected", {"name=outputlayer", ...}));

auto h = conv1(x);     // conv1의 출력 텐서
h = pool1(h);          // pool1의 출력 텐서
auto y = fc(h);        // fc의 출력 텐서

// 단 한 번의 호출로 그래프 추출 + compile + initialize + allocate
model->compile(x, y, ml::train::ExecutionMode::TRAIN);
```

### 4. `compile()` 내부 동작

```cpp
// tensor_api_graph.cpp 내부 로직
int compile(Tensor &input, Tensor &output, ExecutionMode mode) {
  // 1. 출력 텐서에서 역방향 DFS로 그래프 추출
  // 2. 각 텐서의 getProducingLayer() → getInputTensors()를 따라감
  // 3. 리프 텐서 발견 시 자동으로 input 레이어 생성
  // 4. input_layers 연결 문자열 자동 생성
  // 5. 기존 compile(mode) 호출
  // 6. initialize(mode) 호출
  // 7. allocate(mode) 호출
  // 8. API 텐서에 실제 메모리 바인딩
}
```

**장점:**
- `compile(x, y)` 한 번으로 그래프 추출 → 컴파일 → 초기화 → 할당까지 완료
- 텐서 이름에 기반한 자동 input 레이어 생성 (리프 텐서)
- `"input_layers=..."` 문자열을 직접 작성할 필요 없음

### 5. LayerHandle의 operator()

```cpp
class LayerHandle {
  // 단일 입력
  Tensor operator()(const Tensor &input);
  
  // 다중 입력 — mha_core(Q, K, V) 등에 사용
  Tensor operator()(const std::vector<Tensor> &inputs);
  
  // initializer_list 지원 → mha({q, k, v}) 가능
  Tensor operator()(std::initializer_list<Tensor> inputs);
};
```

### 6. 지연 연산 체이닝 (Lazy Operations)

```cpp
auto t = Tensor::ones({1, 1, 1, 1});
t.setValue(0, 0, 0, 0, 10.0f);

// 연산을 큐에 쌓고 eval() 시점에 한 번에 실행
t.chain()
  .add_i(2.0f)        // t += 2
  .multiply_i(3.0f)   // t *= 3
  .eval();             // 결과: (10 + 2) * 3 = 36
```

지원 연산: `add_i`, `subtract_i`, `multiply_i`, `divide_i`, `pow_i`, `inv_sqrt_i`

### 7. 마이그레이션된 샘플 앱 (13개)

| 앱 | 변경 내용 |
|----|------------|
| MNIST | `createGraph()` → `buildGraph()` |
| ResNet | `createResnet18Graph()` → `buildResnet18Graph(Tensor)` |
| SimpleFC | `model->addLayer(...)` → `LayerHandle()()` 체이닝 |
| PicoGPT | 220줄 `addLayer()` → 함수형 체이닝 |
| LLaMA | `createAttentionLayer()` → `Tensor` 반환 |
| YOLOv2/v3 | 블록 함수 → 텐서 체이닝, YOLOv3은 다중 출력 지원 |
| ProductRatings | INI 기반 → `buildGraph()` |
| MixedPrecision, MultiInput, Android/PicoGPTJNI | 동일 패턴 적용 |

---

## 기존 코드와의 비교 예시

### MNIST 모델 빌드

```cpp
// 이전: mnist.ini + main.cpp
model->load(config, MODEL_FORMAT_INI_WITH_BIN);  // INI 파일에 그래프 정의
model->compile();
model->initialize();

// 이후: buildGraph() 함수
static std::pair<Tensor, Tensor> buildGraph() {
  auto x = Tensor({1, 1, 28, 28}, "inputlayer");
  auto h = conv1(x);
  h = pool1(h);
  h = conv2(h);
  h = pool2(h);
  h = flat(h);
  auto y = fc(h);
  return {x, y};
}
auto [x, y] = buildGraph();
model->compile(x, y, TRAIN);
```

---

## 관련 PR

- #3896 이후의 모든 PR(#3898-#3904)은 이 Tensor API에 의존
- 특히 #3902 (CausalLM 마이그레이션)는 Transformer 모델들을 이 API로 전환