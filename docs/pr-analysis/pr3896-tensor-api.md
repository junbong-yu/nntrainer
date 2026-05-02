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

---

## 실제 내부 동작: Syntactic Sugar 검증

겉보기에는 PyTorch 스타일의 함수형 API지만, **내부적으로는 기존 `addLayer()` + `setProperty({"input_layers=..."})` 패턴으로 번역**된다. 아래는 실제 PR diff(`tensor_api_graph.cpp`, `tensor_api_impl.h`)에서 확인한 내용이다.

### 한 문장 요약

> 사용자는 `fc(x)`처럼 PyTorch 스타일로 쓰지만, 내부적으로는 결국 `addLayer(fc, {"input_layers=x"})`로 변환된다.

### 1단계: `fc1(x)` — 아무 연산도 하지 않는다

`LayerHandle::operator()`의 실제 코드 (`tensor_api_graph.cpp` line 5225-5243):

```cpp
Tensor LayerHandle::operator()(const std::vector<Tensor> &inputs) {
  // ❌ 실제 연산을 하지 않음!
  // ❌ 데이터를 이동시키지 않음!

  // ✅ 출력 텐서의 차원만 "추측"함 (fully_connected → output = {batch,1,1,unit})
  TensorDim out_dim = inferOutputDim(ptr_, inputs);

  // ✅ 그래프 엣지(연결 정보)만 기록함
  auto edge = std::make_shared<SymbolicGraphNode>();
  edge->producing_layer = ptr_;              // "이 레이어가 출력을 만들었다"
  edge->dim = out_dim;
  for (auto &inp : inputs) {
    edge->inputs.push_back(inp.impl_->graph_edge);  // "이 입력에서 왔다"
  }

  // ✅ symbolic 출력 텐서 반환 (데이터 없음!)
  output.impl_->graph_edge = edge;
  return output;
}
```

비유하자면 공사 도면에 선을 긋는 것이지, 건물을 짓는 것이 아니다. `x → fc1 → h → fc2 → y`라는 설계도만 그려둔다.

`SymbolicGraphNode` 구조체 (`tensor_api_impl.h` line 5562-5568)도 실제로 `producing_layer`, `inputs`, `dim`, `name`만 가진 경량 구조체다:

```cpp
struct SymbolicGraphNode {
  std::shared_ptr<Layer> producing_layer;              // 생산자 레이어
  std::vector<std::shared_ptr<SymbolicGraphNode>> inputs;  // 입력 엣지들
  TensorDim dim;
  std::string name;
  int output_index = -1;  // split(0) 등 인덱스 출력용
};
```

### 2단계: `model->compile(x, y)` — 여기서 진짜 일이 일어난다

`compile()` 안에서 기존 `addLayer()` 방식으로 번역 (`tensor_api_graph.cpp` line 5266-5513):

```cpp
int Model::compile(std::vector<Tensor> &inputs,
                   std::vector<Tensor> &outputs, ExecutionMode mode) {

  // ┌─────────────────────────────────────────────┐
  // │ 1. DFS: 출력에서 거꾸로 올라가며 레이어 수집  │
  // └─────────────────────────────────────────────┘
  std::function<void(const std::shared_ptr<SymbolicGraphNode> &)> dfs =
    [&](const std::shared_ptr<SymbolicGraphNode> &edge) {
      if (!edge || !edge->producing_layer) return;  // leaf면 멈춤
      if (visited.count(edge->producing_layer.get())) return;
      visited.insert(edge->producing_layer.get());

      for (auto &inp : edge->inputs) {
        dfs(inp);  // ← 입력으로 거꾸로 올라감
      }

      // 이 레이어의 input_layers 이름들을 수집
      std::vector<std::string> input_names;
      for (auto &inp : edge->inputs) {
        if (inp && inp->producing_layer) {
          input_names.push_back(inp->producing_layer->getName());  // "fc1"
        } else {
          input_names.push_back(inp->name);  // "x" (leaf 텐서)
        }
      }
      layers_in_order.push_back({edge->producing_layer, input_names});
    };

  for (auto &output : outputs) {
    dfs(output.impl_->graph_edge);  // 출력 텐서에서 시작
  }

  // ┌─────────────────────────────────────────────┐
  // │ 2. leaf 텐서 → input 레이어 자동 생성         │
  // └─────────────────────────────────────────────┘
  for (auto &inp : inputs) {
    auto input_layer = createLayer("input",
      {"name=" + inp_name, "input_shape=" + shape_str});
    addLayer(std::move(input_layer));  // ← 기존 addLayer() 호출!
  }

  // ┌─────────────────────────────────────────────┐
  // │ 3. 수집한 레이어를 addLayer()로 추가           │
  // │    + input_layers 문자열 설정                  │
  // └─────────────────────────────────────────────┘
  for (auto &info : layers_in_order) {
    if (!info.input_layer_names.empty()) {
      // ⭐ 여기! 문자열로 input_layers를 설정함
      std::string input_layers_str;
      for (size_t i = 0; i < info.input_layer_names.size(); ++i) {
        if (i > 0) input_layers_str += ",";
        input_layers_str += info.input_layer_names[i];
      }
      info.layer->setProperty({"input_layers=" + input_layers_str});
    }
    addLayer(info.layer);  // ← 기존 addLayer() 호출!
  }

  // ┌─────────────────────────────────────────────┐
  // │ 4. 기존 compile() + initialize() + allocate() │
  // └─────────────────────────────────────────────┘
  compile(mode);      // ← 기존 compile() 그대로 호출
  initialize(mode);   // ← 기존 initialize() 그대로 호출
  allocate(mode);     // ← 메모리 할당

  // ┌─────────────────────────────────────────────┐
  // │ 5. API 텐서에 내부 버퍼 바인딩                  │
  // └─────────────────────────────────────────────┘
  for (auto &inp : inputs) {
    inp.impl_->eager_data = std::make_shared<nntrainer::Tensor>(...);
  }
}
```

### 한눈에 보는 변환 과정

```
┌──────────────────────────────────────────────┐
│ 사용자가 쓰는 코드 (PyTorch 스타일)               │
├──────────────────────────────────────────────┤
│ Tensor x({1,1,1,784}, "x");                  │
│ auto h = fc1(x);                             │
│ auto y = fc2(h);                             │
│ model->compile(x, y);                        │
└──────────────────┬───────────────────────────┘
                   │
                   │  compile() 내부에서 번역
                   ▼
┌──────────────────────────────────────────────┐
│ 실제로 호출되는 코드 (기존 방식과 동일)           │
├──────────────────────────────────────────────┤
│ addLayer(createLayer("input",                │
│   {"name=x", "input_shape=1:1:784"}));       │
│                                              │
│ fc1->setProperty({"input_layers=x"});        │
│ addLayer(fc1);                               │
│                                              │
│ fc2->setProperty({"input_layers=fc1"});      │
│ addLayer(fc2);                               │
│                                              │
│ compile(mode);      // 기존 그래프 컴파일       │
│ initialize(mode);   // 기존 초기화             │
│ allocate(mode);     // 기존 메모리 할당         │
└──────────────────────────────────────────────┘
```

### PyTorch와의 결정적 차이

```python
# PyTorch: 이 줄에서 실제로 연산이 일어남
h = fc1(x)  # → 행렬곱이 실행됨, 메모리에 결과가 들어감, autograd 그래프가 생성됨
```

```cpp
// nntrainer PR #3896: 이 줄에서는 아무 일도 안 일어남
auto h = fc1(x);  // → "fc1이 x를 먹는다"는 메모만 적음

// 이 줄에서 비로소 모든 게 실행됨
model->compile(x, y);  // → 그래프 확정, 메모리 할당, 초기화

// 실제 forward 연산은 model->inference()나 model->train() 호출 시 수행됨
```

### 결론

PR #3896은 **사용자 인터페이스(외관)만 PyTorch/Keras 스타일로 바꾼 syntactic sugar**다. `fc1(x)`라고 쓰면 깔끔해 보이지만, 내부적으로는 `compile()`에서 `addLayer()` + `setProperty({"input_layers=..."})`로 번역해서 기존 엔진에 그대로 전달한다. 그래프가 정적으로 컴파일되는 **실행 모델 자체는 전혀 변경되지 않았다**.

---

## 관련 PR

- #3896 이후의 모든 PR(#3898-#3904)은 이 Tensor API에 의존
- 특히 #3902 (CausalLM 마이그레이션)는 Transformer 모델들을 이 API로 전환