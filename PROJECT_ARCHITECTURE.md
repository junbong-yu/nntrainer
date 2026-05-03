# NNTrainer 프로젝트 아키텍처 문서

> NNTrainer: 임베디드 디바이스를 위한 경량 온디바이스 학습/추론 프레임워크

---

## 1. 프로젝트 개요

NNTrainer는 제한된 리소스를 가진 임베디드 디바이스(스마트폰, IoT 기기, Tizen, Android 등)에서 신경망 모델의 **학습(Training)** 과 **추론(Inference)** 을 모두 수행할 수 있도록 설계된 오픈소스 소프트웨어 프레임워크입니다. Apache 2.0 라이선스로 배포되며, Samsung을 중심으로 한 커뮤니티에서 활발히 개발되고 있습니다.

### 1.1 설계 철학

기존의 대규모 딥러닝 프레임워크(TensorFlow, PyTorch 등)는 서버급 하드웨어를 전제로 설계되어, 메모리가 제한된 임베디드 환경에서는 직접 사용하기 어렵습니다. NNTrainer는 다음과 같은 핵심 목표를 가지고 설계되었습니다.

- **온디바이스 학습**: 클라우드에 데이터를 전송하지 않고, 디바이스 자체에서 사용자 데이터로 모델을 파인튜닝하여 개인화를 실현합니다. 전이 학습(Transfer Learning), 퓨샷 학습(Few-Shot Learning), 지속 학습(Continuous Learning) 시나리오를 지원합니다.
- **메모리 효율성**: 제한된 RAM 환경에서도 대형 모델(LLM 포함)을 실행할 수 있도록 FSU(Flash Storage Utilization), MoE Cache, 메모리 플래닝 등 다양한 최적화 기법을 내장합니다.
- **경량 아키텍처**: 불필요한 추상화 계층을 최소화하고, C++로 직접 구현된 핵심 엔진을 통해 오버헤드를 줄입니다.
- **크로스 플랫폼**: Tizen, Android, Ubuntu Linux, Windows, ARM, x86_64, OpenCL GPU 등 다양한 플랫폼과 하드웨어에서 동작합니다.

### 1.2 주요 지원 기능

| 분류 | 지원 내용 |
|------|-----------|
| **모델 아키텍처** | CNN (ResNet, VGG), RNN (LSTM, GRU), Transformer (LLaMA, Qwen, DeepSeek, GPT-OSS), k-NN, 로지스틱 회귀, 강화학습 |
| **학습 방식** | 온디바이스 파인튜닝, 전이 학습, 퓨샷 학습, 동적 학습 최적화(Dynamic Fine-Tuning) |
| **LLM 추론** | FSU(Flash Storage Utilization), MoE(Mixture of Experts) 캐시, 선제적 로딩(Proactive Loading) |
| **데이터 타입** | FP32, FP16, INT4, UINT4, INT8, Q4\_K, BCQ 등 혼합 정밀도(Mixed Precision) 학습 지원 |
| **플랫폼** | Tizen, Android (JNI/NDK), Ubuntu, Windows, OpenCL GPU, Android RPC |
| **API** | C API (공식), C++ API, Java/C# API (예정) |
| **모델 포맷** | ONNX, TFLite, 자체 BIN/INI 포맷 |

---

## 2. 전체 디렉토리 구조

```
nntrainer/
├── nntrainer/                  # 핵심 프레임워크 소스 코드
│   ├── engine.h                # 전역 싱글톤 엔진 (Context, Factory, ThreadPool 관리)
│   ├── context.h               # 실행 컨텍스트 추상화 (AppContext, ClContext)
│   ├── context/*.cpp           # Context 구현체들
│   ├── layers/                 # 레이어 추상화 및 구현
│   │   ├── layer_devel.h       # Layer 추상 기본 클래스
│   │   ├── layer_impl.h        # LayerImpl (가중치/편향 있는 레이어 기반)
│   │   ├── layer_node.h        # LayerNode (그래프 노드 래퍼)
│   │   ├── layer_context.h     # InitLayerContext, RunLayerContext
│   │   └── *.cpp               # 개별 레이어 구현 (convolution, fc, activation 등)
│   ├── graph/                  # 계산 그래프 관리
│   │   ├── graph_core.h        # GraphCore (인접 리스트, 위상 정렬)
│   │   ├── graph_node.h        # GraphNode 추상 인터페이스
│   │   └── network_graph.h     # NetworkGraph (레이어 노드 컨테이너 + Manager)
│   ├── tensor/                 # 텐서 및 메모리 관리
│   │   ├── tensor.h            # Tensor 클래스 (Pimpl 패턴, 다중 데이터타입)
│   │   ├── tensor_base.h       # TensorBase 추상 클래스 (FloatTensor, HalfTensor 등)
│   │   ├── var_grad.h          # Var_Grad (변수+그래디언트 쌍)
│   │   ├── weight.h            # Weight (Var_Grad 확장, 정규화/옵티마이저 변수)
│   │   ├── manager.h           # Manager (중앙 텐서 코디네이터)
│   │   ├── tensor_pool.h       # TensorPool (풀드 할당기 + MemoryPlanner)
│   │   └── memory_pool.h       # MemoryPool (물리적 메모리: mmap, rpcmem)
│   ├── models/                 # 모델 진입점
│   │   └── neuralnet.h         # NeuralNetwork 클래스 (학습/추론 메인 클래스)
│   ├── optimizers/             # 최적화기
│   │   ├── optimizer_devel.h   # Optimizer 추상 클래스
│   │   ├── optimizer_context.h # RunOptimizerContext
│   │   └── *.cpp               # Adam, AdamW, SGD 등 구현
│   └── common/                 # 공통 유틸리티
├── api/                        # 공개 API
│   ├── capi/                   # C API (nntrainer.h) - Tizen 공식
│   └── ccapi/                  # C++ API - 기타 플랫폼
├── jni/                        # Android JNI 바인딩
├── Applications/               # 예제 애플리케이션
│   ├── CausalLM/               # LLM 추론 예제 (Qwen3, GPT-OSS 등)
│   └── ...                     # ResNet, VGG, Few-shot 등 다양한 예제
├── test/                       # 단위 테스트
├── tools/                      # 도구 및 유틸리티
├── nnstreamer/                 # NNStreamer 통합
├── benchmarks/                 # 벤치마크
├── docs/                       # 문서
├── packaging/                  # 패키징 스크립트 (RPM, DEB 등)
├── debian/                     # Debian 패키징
├── meson.build                 # Meson 빌드 시스템
└── README.md                   # 프로젝트 개요
```

### 2.1 주요 디렉토리 역할

| 디렉토리 | 역할 |
|----------|------|
| `nntrainer/` | 프레임워크의 핵심 구현. 엔진, 컨텍스트, 레이어, 텐서, 그래프, 옵티마이저 등 모든 내부 컴포넌트가 위치 |
| `api/capi/` | C 언어 공개 API. Tizen 플랫폼에서 공식으로 사용하는 인터페이스 |
| `api/ccapi/` | C++ 공개 API. Android, Linux, Windows 등 기타 플랫폼에서 사용 |
| `jni/` | Android NDK/JNI 바인딩. Java/Kotlin에서 NNTrainer를 호출할 수 있도록 연결 |
| `Applications/` | 실제 사용 예제. LLM 추론(CausalLM), 이미지 분류, 개인화 학습 등 다양한 시나리오 |
| `test/` | 단위 테스트 및 통합 테스트. Google Test 기반 |
| `tools/` | 모델 변환, 프로파일링, 디버깅 도구 |

---

## 3. 핵심 아키텍처 개념

NNTrainer의 아키텍처는 **Engine - Context - NeuralNetwork - NetworkGraph - LayerNode - Layer** 의 계층적 구조로 이루어져 있습니다. 각 계층은 명확한 책임을 가지며, 느슨한 결합을 유지합니다.

### 3.1 Engine - 전역 싱글톤

`Engine` 클래스는 NNTrainer의 **전역 싱글톤** 으로, 프레임워크 전체의 생명주기와 리소스를 관리하는 중앙 허브입니다. `Singleton<Engine>` 템플릿을 상속받아 단 하나의 인스턴스만 존재함이 보장됩니다.

```cpp
// nntrainer/engine.h
class Engine : public Singleton<Engine> {
  // ...
  std::unordered_map<std::string, nntrainer::Context *> engines;
  std::unordered_map<std::string, std::shared_ptr<nntrainer::MemAllocator>> allocator;
  std::unique_ptr<ThreadPoolManager> thread_pool_manager_;
};
```

#### 주요 책임

1. **Context 등록 및 관리**: 최대 16개의 Context를 `engines` 맵에 등록합니다. 이름 기반 조회가 가능하며, 스레드 세이프한 뮤텍스(`engine_mutex`)로 보호됩니다.

2. **플러그인 로딩**: `registerContext(library_path)` 메서드를 통해 공유 라이브러리(`.so`)를 동적으로 로딩합니다. 플러그인은 `extern "C" ContextPluggable ml_train_context_pluggable` 심볼을 반드시 정의해야 합니다.

3. **팩토리 메서드**: `createLayerObject()`, `createOptimizerObject()`, `createLearningRateSchedulerObject()` 메서드를 제공합니다. 내부적으로 등록된 Context의 팩토리에 위임하여 객체를 생성합니다.

4. **ThreadPool 관리**: `getThreadPoolManager()`를 통해 스레드 풀 매니저에 접근합니다. 지연 초기화(lazy initialization) 패턴을 사용하며, 뮤텍스로 스레드 안전성을 보장합니다.

5. **Compute Engine 파싱**: 레이어 속성에서 `engine=cpu` 또는 `engine=cl` 같은 키워드를 파싱하여 적절한 Context를 선택합니다.

#### Engine과 Context의 관계

```
┌─────────────────────────────────────────────┐
│                  Engine (Singleton)          │
│  ┌───────────────────────────────────────┐  │
│  │  engines: {                           │  │
│  │    "app"  → AppContext* (CPU 기본)     │  │
│  │    "cl"   → ClContext* (OpenCL GPU)   │  │
│  │    ...    → (플러그인 Context)         │  │
│  │  }                                    │  │
│  └───────────────────────────────────────┘  │
│  ┌───────────────────────────────────────┐  │
│  │  allocator: {                         │  │
│  │    "app"  → MemAllocator              │  │
│  │    "cl"   → MemAllocator              │  │
│  │  }                                    │  │
│  └───────────────────────────────────────┘  │
│  ┌───────────────────────────────────────┐  │
│  │  thread_pool_manager_ (지연 초기화)    │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### 3.2 Context - 실행 컨텍스트 추상화

`Context`는 하드웨어 백엔드(CPU, GPU, NPU 등)에 따른 실행 환경을 추상화하는 **추상 기본 클래스** 입니다. 각 Context는 고유한 메모리 할당기, 레이어 팩토리, 옵티마이저 팩토리를 보유합니다.

```cpp
// nntrainer/context.h
class Context {
public:
  virtual int init() { return 0; }
  virtual PtrType<nntrainer::Layer> createLayerObject(const std::string &type, ...);
  virtual PtrType<nntrainer::Optimizer> createOptimizerObject(const std::string &type, ...);
  virtual std::string getName() = 0;
  std::shared_ptr<MemAllocator> getMemAllocator();
};
```

#### ContextData

각 Context는 `ContextData`를 통해 실행 중 생성된 컨텍스트별 데이터(메모리 할당기 등)를 관리합니다. `RunLayerContext`는 이 `ContextData`를 참조하여 레이어 실행에 필요한 리소스에 접근합니다.

#### 주요 구현체

| 구현체 | 설명 |
|--------|------|
| **AppContext** | 기본 CPU 실행 컨텍스트. 모든 플랫폼에서 사용 가능한 기본 백엔드 |
| **ClContext** | OpenCL GPU 실행 컨텍스트. `ENABLE_OPENCL` 빌드 옵션 활성화 시 사용 |

#### 플러그인 시스템

Context는 공유 라이브러리를 통한 동적 확장을 지원합니다. 플러그인 라이브러리는 다음 구조체를 정의해야 합니다.

```cpp
typedef struct {
  CreateContextFunc createfunc;   // Context 생성 함수
  DestroyContextFunc destroyfunc; // Context 소멸 함수
} ContextPluggable;

extern "C" ContextPluggable ml_train_context_pluggable;
```

`Engine::registerContext()`는 `dlopen()`으로 라이브러리를 로딩하고, `ml_train_context_pluggable` 심볼을 찾아 Context를 등록합니다.

### 3.3 NeuralNetwork (Model) - 모델 진입점

`NeuralNetwork` 클래스는 NNTrainer에서 모델을 다루는 **최상위 진입점** 입니다. `ml::train::Model` 인터페이스를 구현하며, 사용자는 이 클래스를 통해 모델의 생성, 컴파일, 학습, 추론, 저장, 로딩을 수행합니다.

```cpp
// nntrainer/models/neuralnet.h
class NeuralNetwork : public ml::train::Model {
  NetworkGraph model_graph;           // 네트워크 계산 그래프
  std::shared_ptr<OptimizerWrapped> opt; // 옵티마이저 (각 레이어에 복사됨)
  std::array<std::shared_ptr<DataBuffer>, 3> data_buffers; // 데이터 버퍼
  const Engine *ct_engine;            // 바인딩된 엔진
  bool compiled;                      // 컴파일 여부
  bool initialized;                   // 초기화 여부
  // ...
};
```

#### 주요 구성 요소와의 관계

- **NetworkGraph**: 레이어들의 연결 관계와 실행 순서를 관리하는 계산 그래프
- **OptimizerWrapped**: 옵티마이저를 감싼 래퍼. 각 레이어의 가중치 업데이트에 사용
- **DataBuffer[]**: 학습/검증/테스트 데이터셋을 공급하는 버퍼 (3개: train, valid, test)
- **Engine**: 객체 생성과 컨텍스트 관리를 위임하는 전역 엔진

#### 생명주기: compile → initialize → train

```
1. 생성 (new NeuralNetwork)
   └→ 빈 모델 상태

2. loadFromConfig(config_path)
   └→ INI 파일에서 모델 구조, 레이어, 하이퍼파라미터 로드

3. compile(ExecutionMode::TRAIN)
   └→ 그래프 검증, 위상 정렬, 실행 순서 설정, Loss 레이어 추가
   └→ compiled = true

4. initialize(ExecutionMode::TRAIN)
   └→ 텐서 메모리 할당, 가중치 초기화, 옵티마이저 변수 생성
   └→ initialized = true

5. train() / inference()
   └→ 실제 학습 또는 추론 실행
```

### 3.4 NetworkGraph - 계산 그래프

`NetworkGraph`는 신경망을 구성하는 모든 `LayerNode`를 담고 있는 **그래프 컨테이너** 입니다. 내부적으로 `GraphCore`를 사용하여 위상 정렬된 실행 순서를 관리하고, `Manager`를 통해 텐서 메모리를 통합 관리합니다.

```cpp
// nntrainer/graph/network_graph.h
class NetworkGraph {
  std::shared_ptr<Manager> tensor_manager;  // 중앙 텐서 코디네이터
  GraphCore graph;                          // 코어 그래프 (위상 정렬)
  bool compiled;                            // 컴파일 완료 여부
  unsigned int batch_size;                  // 현재 배치 크기
  LayerNode *backward_iter_end;             // 역전파 종료 노드 (메모리 최적화)
  LayerNode *forward_iter_end;              // 순전파 종료 노드
  bool optimize_memory;                     // 메모리 최적화 활성화
  std::string tensor_format;                // NCHW 또는 NHWC
  std::vector<std::string> tensor_dtype;    // 가중치 타입 - 활성화 타입 (예: "FP32-FP16")
  // ...
};
```

#### GraphCore - 그래프 코어

`GraphCore`는 인접 리스트(adjacency list) 기반의 그래프 자료구조를 제공합니다.

```cpp
// nntrainer/graph/graph_core.h
class GraphCore {
  std::vector<std::shared_ptr<GraphNode>> node_list;  // 정렬 전 노드 리스트
  std::vector<std::shared_ptr<GraphNode>> Sorted;     // 위상 정렬된 노드 리스트
  std::unordered_map<std::string, int> node_map;      // 이름 → 인덱스 맵
  std::unordered_map<std::string, int> sorted_node_map; // 정렬 후 이름 → 인덱스

  void topologicalSort();  // DFS 기반 위상 정렬
  void makeAdjacencyList(...); // 인접 리스트 생성
};
```

#### 위상 정렬과 실행 순서

`NetworkGraph::compile()` 단계에서 `GraphCore::topologicalSort()`가 호출되어 레이어들을 의존성 순서대로 정렬합니다. 이후 `setExecutionOrder()`에서 각 노드에 4단계 실행 순서가 할당됩니다.

```cpp
// nntrainer/graph/graph_node.h
class GraphNode {
  // ExecutionOrder: (forward_order, calc_gradient_order, calc_derivative_order, apply_gradient_order)
  typedef std::tuple<unsigned int, unsigned int, unsigned int, unsigned int> ExecutionOrder;
};
```

- **forward_order**: 순전파 실행 순서 (위상 정렬 순서와 동일)
- **calc_gradient_order**: 가중치 그래디언트 계산 순서
- **calc_derivative_order**: 입력에 대한 미분 계산 순서 (이전 레이어로 전달)
- **apply_gradient_order**: 옵티마이저를 통한 가중치 업데이트 순서

#### Manager를 통한 텐서 메모리 관리

`NetworkGraph`는 `Manager`를 멤버로 보유하며, 모든 텐서(입력, 출력, 가중치, 중간 텐서)의 생명주기를 관리합니다.

```
NetworkGraph
  ├── GraphCore (위상 정렬된 LayerNode[])
  └── Manager
        ├── weight_pool (TensorPool) - 가중치 풀
        ├── tensor_pool (TensorPool) - 활성화/중간 텐서 풀
        └── MemoryPlanner (메모리 레이아웃 계획)
```

### 3.5 Layer / LayerNode - 레이어 추상화

NNTrainer의 레이어 시스템은 **3단계 계층** 으로 설계되어 있습니다.

#### 3.5.1 Layer (layer_devel.h) - 추상 기본 클래스

`Layer`는 모든 레이어의 최상위 추상 클래스입니다. 순전파, 역전파, 속성 설정 등의 핵심 인터페이스를 정의합니다.

```cpp
// nntrainer/layers/layer_devel.h
class Layer {
public:
  virtual const std::string getType() const = 0;
  virtual void finalize(InitLayerContext &context) = 0;
  virtual void initialize(RunLayerContext &context) {}
  virtual void forwarding(RunLayerContext &context, bool training) = 0;
  virtual void calcDerivative(RunLayerContext &context) = 0;
  virtual void calcGradient(RunLayerContext &context) {}
  virtual void setProperty(const std::vector<std::string> &values) = 0;
  virtual bool supportBackwarding() const = 0;
  virtual bool supportInPlace() const { return is_inplace; }
  // ...
};
```

#### 3.5.2 LayerImpl (layer_impl.h) - 가중치/편향 기반 레이어

`LayerImpl`은 `Layer`를 상속하며, 가중치(weight)와 편향(bias)을 갖는 레이어들을 위한 공통 기반을 제공합니다. 정규화(regularizer), 초기화(initializer), 가중치 감쇠(weight decay) 등의 속성 파싱을 처리합니다.

```cpp
// nntrainer/layers/layer_impl.h
class LayerImpl : public virtual Layer {
protected:
  std::unique_ptr<std::tuple<
    props::WeightRegularizer,
    props::WeightRegularizerConstant,
    props::WeightInitializer,
    props::WeightDecay,
    props::BiasDecay,
    props::BiasInitializer,
    props::DisableBias,
    props::Print
  >> layer_impl_props;
};
```

#### 3.5.3 LayerNode (layer_node.h) - 그래프 노드 래퍼

`LayerNode`는 `Layer` 객체를 감싸서 `GraphNode` 인터페이스를 구현하는 **래퍼 클래스** 입니다. 그래프 연결 정보(입력/출력 커넥션), 실행 순서, In-Place 최적화 여부 등을 관리합니다.

```cpp
// nntrainer/layers/layer_node.h
class LayerNode final : public ml::train::Layer, public GraphNode {
  std::unique_ptr<nntrainer::Layer> layer;        // 실제 레이어 객체
  std::unique_ptr<RunLayerContext> run_context;   // 실행 컨텍스트
  ExecutionOrder exec_order;                      // 실행 순서
  InPlaceType inplace_type;                       // In-Place 모드
  std::vector<std::unique_ptr<Connection>> output_connections; // 출력 연결
  // ...
};
```

#### 3.5.4 InitLayerContext vs RunLayerContext

레이어는 두 가지 컨텍스트를 통해 프레임워크와 상호작용합니다.

| 컨텍스트 | 시점 | 역할 |
|----------|------|------|
| **InitLayerContext** | `finalize()` 호출 시 | 레이어 초기화. 입력 차원 제공, 출력 차원 설정, 가중치/텐서 요청 (`requestWeight()`, `requestTensor()`) |
| **RunLayerContext** | `forwarding()`, `calcDerivative()`, `calcGradient()` 호출 시 | 레이어 실행. 가중치/입력/출력/그래디언트 접근 (`getWeight()`, `getInput()`, `getOutput()`, `getOutputGrad()`) |

```cpp
// InitLayerContext - 초기화 시점
unsigned int requestWeight(const TensorDim &dim, const Initializer init, ...);
unsigned int requestTensor(const TensorDim &dim, const std::string &name, ...);
void setOutputDimensions(const std::vector<TensorDim> &out_dim);

// RunLayerContext - 실행 시점
Tensor &getWeight(unsigned int idx);
Tensor &getWeightGrad(unsigned int idx);
Tensor &getInput(unsigned int idx);
Tensor &getInputGrad(unsigned int idx);
Tensor &getOutput(unsigned int idx);
const Tensor getOutputGrad(unsigned int idx);
Tensor &getTensor(unsigned int idx);
```

#### 3.5.5 forward/backward 메서드 시그니처와 호출 흐름

```
Layer::forwarding(RunLayerContext &context, bool training)
  → 입력 텐서 읽기 → 연산 수행 → 출력 텐서 쓰기

Layer::calcDerivative(RunLayerContext &context)
  → 출력 그래디언트 읽기 → 입력에 대한 미분 계산 → 입력 그래디언트 쓰기
  → 이전 레이어로 전달할 미분 계산

Layer::calcGradient(RunLayerContext &context)
  → 출력 그래디언트 읽기 → 가중치에 대한 그래디언트 계산 → 가중치 그래디언트 쓰기
  → 옵티마이저가 사용할 그래디언트 계산
```

### 3.6 Tensor / Var_Grad / Manager - 텐서 관리

#### 3.6.1 Tensor 클래스

`Tensor`는 다차원 행렬을 표현하는 핵심 데이터 구조입니다. **Pimpl 패턴** 을 사용하여 내부 구현(`TensorBase`)을 숨기고, 다양한 데이터 타입을 투명하게 지원합니다.

```cpp
// nntrainer/tensor/tensor.h
class Tensor {
  std::unique_ptr<TensorBase> itensor_;  // 실제 데이터 저장 (Pimpl)
  // ...
};
```

**지원 데이터 타입**:
- `FP32`: 단정밀도 부동소수점 (기본)
- `FP16`: 반정밀도 부동소수점 (`ENABLE_FP16` 빌드 시)
- `INT8`, `INT16`, `INT32`: 정수 양자화
- `UINT4`, `INT4`: 4비트 양자화
- `Q4_0`, `Q4_K`: GGUF 스타일 4비트 양자화
- `BCQ`: Binary Code Quantization

**지원 메모리 포맷**:
- `NCHW`: 채널 우선 (기본)
- `NHWC`: 채널 마지막

#### 3.6.2 Var_Grad (변수+그래디언트 쌍)

`Var_Grad`는 학습 가능한 변수(tensor)와 해당 그래디언트(gradient)를 쌍으로 관리하는 클래스입니다.

```cpp
// nntrainer/tensor/var_grad.h
class Var_Grad {
  std::shared_ptr<Tensor> var;   // 변수 텐서
  std::shared_ptr<Tensor> grad;  // 그래디언트 텐서
  bool is_dependent;             // 다른 가중치와 공유 여부
  bool is_first_access_gradient; // 첫 그래디언트 접근 여부
  bool is_last_access_gradient;  // 마지막 그래디언트 접근 여부
};
```

#### 3.6.3 Weight (경량 뷰)

`Weight`는 `Var_Grad`를 확장하여, 정규화(regularization), 옵티마이저 변수(optimizer variables), 혼합 정밀도 마스터 가중치(FP32) 등의 기능을 추가합니다.

```cpp
// nntrainer/tensor/weight.h
class Weight : public Var_Grad {
  WeightRegularizer regularizer;     // 정규화 타입 (L2Norm 등)
  float regularizer_constant;        // 정규화 상수
  float decay;                       // 가중치 감쇠
  float clip_by_global_norm;         // 그래디언트 클리핑
  std::vector<Tensor *> opt_vars;    // 옵티마이저 변수 (모멘텀 등)
  std::shared_ptr<Tensor> var32;     // FP32 마스터 가중치 (혼합 정밀도용)
  bool is_mixed;                     // 혼합 정밀도 여부
  float loss_scale;                  // 손실 스케일링
};
```

#### 3.6.4 Manager - 중앙 텐서 코디네이터

`Manager`는 네트워크 내 모든 텐서의 요청, 할당, 해제를 중앙에서 관리하는 코디네이터입니다. `TensorPool`을 통해 메모리를 효율적으로 재사용합니다.

```cpp
// nntrainer/tensor/manager.h
class Manager {
  TensorPool weight_pool;   // 가중치 전용 풀
  TensorPool tensor_pool;   // 활성화/중간 텐서 풀
  bool enable_fsu;          // FSU 활성화 여부
  unsigned int fsu_lookahead; // FSU 선제적 로딩 크기
  // ...
};
```

#### 3.6.5 TensorPool과 MemoryPool

`TensorPool`은 풀드 할당기(pooled allocator)로, `MemoryPlanner`와 협력하여 텐서들의 메모리 레이아웃을 계획합니다. 서로 생명주기가 겹치지 않는 텐서들은 동일한 물리적 메모리를 공유할 수 있습니다.

```
TensorPool
  ├── BasicPlanner (메모리 레이아웃 계획 알고리즘)
  └── MemoryPool (물리적 메모리 할당)
        ├── MMapedMemory (mmap 기반 메모리)
        ├── RpcMem (Android rpcmem 할당)
        └── StandardMemory (일반 malloc/new)
```

#### 3.6.6 TensorLifespan 개념

텐서의 생명주기는 `TensorLifespan` 열거형으로 정의되며, 메모리 재사용 전략에 직접적인 영향을 미칩니다.

| Lifespan | 설명 |
|----------|------|
| `FORWARD_FUNCTION_LIFESPAN` | 순전파 함수 내에서만 유효. 가장 짧은 생명주기 |
| `CALC_GRAD_DERIV_LIFESPAN` | 그래디언트/미분 계산 동안 유효 |
| `ITERATION_LIFESPAN` | 단일 학습 반복(iteration) 동안 유효 |
| `EPOCH_LIFESPAN` | 전체 에포크 동안 유효 |

### 3.7 Optimizer - 최적화기

`Optimizer`는 가중치 업데이트 규칙을 정의하는 추상 클래스입니다.

```cpp
// nntrainer/optimizers/optimizer_devel.h
class Optimizer {
public:
  virtual double getDefaultLearningRate() const = 0;
  virtual void applyGradient(RunOptimizerContext &context) = 0;
  virtual void setProperty(const std::vector<std::string> &values);
  virtual std::vector<TensorDim> getOptimizerVariableDim(const TensorDim &dim) = 0;
  virtual const std::string getType() const = 0;
};
```

#### RunOptimizerContext

옵티마이저 실행 시 전달되는 컨텍스트로, 가중치, 그래디언트, 학습률, 현재 반복 횟수에 접근할 수 있습니다.

```cpp
// nntrainer/optimizers/optimizer_context.h
class RunOptimizerContext {
  Weight *weight;       // 대상 가중치
  size_t iteration;     // 현재 반복 횟수
  double learning_rate; // 현재 학습률

  Tensor &getWeight();
  Tensor &getGradient();
  Tensor &getWeightFP32();       // FP32 마스터 가중치 (혼합 정밀도)
  Tensor &getOptimizerVariable(unsigned int idx); // 모멘텀 등
  void applyGradient(double lr);
};
```

#### 주요 구현체

| 옵티마이저 | 설명 |
|------------|------|
| **SGD** | 확률적 경사 하강법. 모멘텀 지원 |
| **Adam** | Adaptive Moment Estimation. 기본 베타 값(0.9, 0.999) |
| **AdamW** | Adam + Weight Decay 분리. L2 정규화와 가중치 감쇠를 분리 |

---

## 4. 실행 모델

### 4.1 순방향 전파 (Forward Propagation)

순방향 전파는 `NeuralNetwork::forwarding()` 에서 시작하여 `NetworkGraph`를 거쳐 각 `LayerNode`의 `forwarding()` 이 순차적으로 호출되는 구조입니다.

```
NeuralNetwork::forwarding(training)
  └→ NetworkGraph::forwarding(training, forwarding_op, stop_cb)
        └→ GraphCore의 위상 정렬된 노드 순회 (cbegin → cend)
              └→ LayerNode::forwarding(training)
                    └→ layer->forwarding(run_context, training)
                          └→ RunLayerContext에서 입력 읽기 → 연산 → 출력 쓰기
```

**증분 순전파(Incremental Forwarding)**: LLM 추론에서 시퀀스의 일부 구간(`from` ~ `to`)만 실행할 수 있는 `incremental_forwarding()` 도 지원합니다. 이는 KV 캐시와 결합하여 토큰 생성 시 매 단계 전체 네트워크를 재실행하지 않도록 합니다.

```cpp
// NeuralNetwork
sharedConstTensors incremental_forwarding(
  unsigned int from, unsigned int to,
  sharedConstTensors input, sharedConstTensors label,
  bool training);
```

### 4.2 역방향 전파 (Backward Propagation)

역방향 전파는 두 단계로 분리되어 있습니다.

```
NeuralNetwork::backwarding(iteration)
  └→ NetworkGraph::backwarding(iteration, forwarding_op, backwarding_op, lazy_apply_grad_op)
        └→ 역방향 위상 정렬 순서로 순회 (crbegin → crend)
              └→ LayerNode::calcDerivative()
                    └→ layer->calcDerivative(run_context)
                    └→ 출력 그래디언트 → 입력 미분 계산 → 이전 레이어로 전달
              └→ LayerNode::calcGradient()
                    └→ layer->calcGradient(run_context)
                    └→ 출력 그래디언트 → 가중치 그래디언트 계산
              └→ NetworkGraph::applyGradients()
                    └→ 옵티마이저를 통한 가중치 업데이트
```

#### calcDerivative vs calcGradient 분리의 의미

| 메서드 | 목적 | 출력 |
|--------|------|------|
| **calcDerivative** | 이전 레이어로 전달할 입력 미분 계산 | `input_grad` 텐서에 저장 |
| **calcGradient** | 현재 레이어의 가중치 업데이트용 그래디언트 계산 | `weight_grad` 텐서에 저장 |

이 분리는 다음과 같은 이점을 제공합니다.

1. **메모리 최적화**: `calcDerivative`가 필요 없는 레이어(예: Loss 레이어)는 해당 단계를 건너뛸 수 있습니다.
2. **그래디언트 클리핑**: `calcGradient` 결과를 즉시 적용하지 않고, 전역 노름(global norm) 계산 후 일괄 적용할 수 있습니다.
3. **지연 적용(Lazy Apply)**: 혼합 정밀도 학습에서 손실 스케일링(loss scaling)과 결합하여 그래디언트 오버플로우를 방지합니다.

### 4.3 학습 루프

학습은 `NeuralNetwork::train()` → `train_run()` 의 흐름으로 실행되며, 에포크 루프 내에서 forward → backward → optimizer apply 순서가 반복됩니다.

```
NeuralNetwork::train(values, stop_cb, epoch_complete_cb)
  └→ train_run(stop_cb, epoch_complete_cb)
        └→ for epoch = 0 to epochs:
              └→ for each batch in dataset:
                    │
                    │  1. Forward Propagation
                    │  └→ forwarding(training=true)
                    │     └→ NetworkGraph::forwarding()
                    │           └→ 각 LayerNode::forwarding()
                    │
                    │  2. Loss 계산
                    │  └→ Loss 레이어에서 출력과 레이블 비교
                    │
                    │  3. Backward Propagation
                    │  └→ backwarding(iteration)
                    │     └→ 각 LayerNode::calcDerivative() (역순)
                    │     └→ 각 LayerNode::calcGradient() (역순)
                    │
                    │  4. Optimizer Apply
                    │  └→ NetworkGraph::applyGradients()
                    │     └→ 각 가중치에 대해 optimizer->applyGradient()
                    │
                    └→ Dynamic Training Optimization (선택적)
                          └→ 그래디언트 노름이 임계값 이하면 업데이트 스킵
              └→ epoch_complete_cb() 호출
              └→ 검증 데이터로 validation loss 계산
```

---

## 5. 데이터 흐름

### 5.1 Training Flow (컴파일 → 초기화 → 에포크 루프)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Training Flow                                │
│                                                                     │
│  [사용자]                                                           │
│     │                                                               │
│     ▼                                                               │
│  ┌──────────────┐                                                   │
│  │ NeuralNetwork│                                                   │
│  │   생성        │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │ loadFromConfig│  ← INI 파일에서 모델 정의 로드                     │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │   compile()   │  ← 그래프 검증, 위상 정렬, 실행 순서 설정          │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │ initialize()  │  ← 텐서 메모리 할당, 가중치 초기화                │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Epoch Loop                                │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │                  Batch Loop                            │  │  │
│  │  │                                                        │  │  │
│  │  │   ┌──────────┐    ┌──────────┐    ┌──────────────┐   │  │  │
│  │  │   │ Forward  │───▶│  Loss    │───▶│  Backward    │   │  │  │
│  │  │   │ Prop.    │    │  계산     │    │  Prop.       │   │  │  │
│  │  │   └──────────┘    └──────────┘    └──────┬───────┘   │  │  │
│  │  │                                          │           │  │  │
│  │  │                                          ▼           │  │  │
│  │  │                                  ┌──────────────┐   │  │  │
│  │  │                                  │ Optimizer    │   │  │  │
│  │  │                                  │ Apply Grad   │   │  │  │
│  │  │                                  └──────────────┘   │  │  │
│  │  │                                                      │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                          │                                 │  │
│  │                          ▼                                 │  │
│  │                  ┌──────────────┐                          │  │
│  │                  │ Validation   │                          │  │
│  │                  │ Loss 계산     │                          │  │
│  │                  └──────────────┘                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Forward Tensor Flow (입력 → 레이어 → 출력)

```
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐
│  Input   │────▶│  Conv2D  │────▶│   ReLU   │────▶│   FC     │────▶│  Output  │
│ Tensor   │     │  Layer   │     │  Layer   │     │  Layer   │     │ (Logits) │
└─────────┘     └──────────┘     └──────────┘     └──────────┘     └─────────┘
                    │                                  │
                    ▼                                  ▼
              ┌──────────┐                       ┌──────────┐
              │  Weight   │                       │  Weight   │
              │  + Bias   │                       │  + Bias   │
              └──────────┘                       └──────────┘

  각 LayerNode::forwarding():
    input = context.getInput(0)          ← 이전 레이어 출력 (또는 모델 입력)
    weight = context.getWeight(0)        ← 가중치 텐서
    output = context.getOutput(0)        ← 출력 텐서 (새로 계산)
    output = operation(input, weight)    ← 실제 연산
```

### 5.3 Backward Gradient Flow (Loss → 레이어 → 가중치)

```
  ┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  Loss    │◀────│  FC Layer    │◀────│  ReLU Layer  │◀────│  Conv2D Layer│
  │ Gradient │     │  calcDeriv   │     │  calcDeriv   │     │  calcDeriv   │
  └─────────┘     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
                         │                    │                    │
                         ▼                    ▼                    ▼
                   ┌──────────┐         ┌──────────┐         ┌──────────┐
                   │  Weight   │         │  (N/A)   │         │  Weight   │
                   │  calcGrad │         │          │         │  calcGrad │
                   └──────────┘         └──────────┘         └──────────┘
                         │                                        │
                         ▼                                        ▼
                   ┌──────────┐                           ┌──────────┐
                   │  Adam    │                           │  Adam    │
                   │  Apply   │                           │  Apply   │
                   └──────────┘                           └──────────┘

  역방향 순서 (위상 정렬의 역순):
    1. Loss Layer: dL/doutput 계산
    2. FC Layer:
       - calcDerivative: dL/dinput 계산 → ReLU로 전달
       - calcGradient: dL/dweight 계산 → 옵티마이저로 전달
    3. ReLU Layer:
       - calcDerivative: dL/dinput 계산 → Conv2D로 전달
       - calcGradient: 없음 (가중치 없음)
    4. Conv2D Layer:
       - calcDerivative: dL/dinput 계산 (입력 레이어이므로 버림)
       - calcGradient: dL/dweight 계산 → 옵티마이저로 전달
    5. 모든 가중치에 대해 Optimizer::applyGradient() 호출
```

---

## 6. 설계 패턴

NNTrainer는 여러 고전적 설계 패턴을 활용하여 확장성과 유지보수성을 확보하고 있습니다.

### 6.1 Factory Pattern (레이어/옵티마이저 생성)

`Engine`과 `Context`는 팩토리 메서드를 통해 레이어와 옵티마이저를 생성합니다. 문자열 타입 이름 또는 정수 키로 객체를 생성할 수 있습니다.

```cpp
// Engine을 통한 팩토리 호출
std::unique_ptr<Layer> layer = engine->createLayerObject("convolution2d", props);
std::unique_ptr<Optimizer> opt = engine->createOptimizerObject("adam", props);

// 내부적으로 Context의 팩토리에 위임
// ct->createLayerObject(type) → 등록된 FactoryType<T> 호출
```

플러그인 레이어는 `extern "C" LayerPluggable ml_train_layer_pluggable` 구조체를 통해 팩토리에 등록됩니다.

### 6.2 Singleton Pattern (Engine, AppContext)

`Engine`은 `Singleton<Engine>` 템플릿을 상속받아 전역적으로 단일 인스턴스를 보장합니다.

```cpp
template <typename T>
class Singleton {
protected:
  static T* instance;
  static std::once_flag init_flag;

  static T& getInstance() {
    std::call_once(init_flag, []() {
      instance = new T();
      instance->initialize();
    });
    return *instance;
  }
};
```

### 6.3 Strategy Pattern (MemoryPlanner)

`TensorPool`은 `MemoryPlanner` 인터페이스를 통해 메모리 할당 전략을 교체할 수 있습니다. `BasicPlanner`가 기본 구현체이며, 실행 순서 분석을 통해 텐서들의 생명주기를 파악하고 메모리를 재사용합니다.

```
TensorPool
  ├── setPlanner(std::unique_ptr<MemoryPlanner>)  ← 전략 교체
  ├── BasicPlanner                                 ← 기본 전략
  └── plan()                                       ← 메모리 레이아웃 계산
```

### 6.4 Plugin Pattern (공유 라이브러리)

Context, Layer, Optimizer는 모두 공유 라이브러리(`.so`)를 통한 동적 로딩을 지원합니다.

```cpp
// 플러그인 구조체
typedef struct {
  CreateLayerFunc createfunc;
  DestroyLayerFunc destroyfunc;
} LayerPluggable;

extern "C" LayerPluggable ml_train_layer_pluggable;
```

`Engine::registerContext(library_path)`는 `dlopen()`/`dlsym()`을 사용하여 런타임에 플러그인을 로딩합니다.

### 6.5 Observer Pattern (학습 콜백)

`NeuralNetwork::train()` 은 콜백 함수를 인자로 받아 학습 진행 상황을 외부에 알립니다.

```cpp
int train(
  const std::vector<std::string> &values,
  std::function<bool(void *)> stop_cb,           // 학습 중단 여부 확인
  void *stop_user_data,
  std::function<void(void *)> epoch_complete_cb, // 에포크 완료 시 호출
  void *epoch_user_data
);
```

### 6.6 Pimpl Pattern (Tensor)

`Tensor` 클래스는 `std::unique_ptr<TensorBase>`를 내부 멤버로 보유하여, 실제 데이터 타입(FP32, FP16, INT8 등)에 따른 구현을 숨깁니다.

```cpp
class Tensor {
  std::unique_ptr<TensorBase> itensor_;  // 실제 구현 (FloatTensor, HalfTensor 등)
public:
  template <typename T = float> T *getData() const {
    return (T *)itensor_->getData();
  }
};
```

이 패턴의 장점은 다음과 같습니다.

- 헤더 파일에 템플릿 구현이 노출되지 않아 컴파일 시간 단축
- 새로운 데이터 타입 추가 시 `Tensor` 인터페이스 변경 불필요
- 런타임에 데이터 타입 결정 가능

---

## 7. 특별한 기능

### 7.1 FSU (Flash Storage Utilization) - LLM 추론용

FSU는 플래시 저장소를 활용하여 모델 가중치를 필요할 때마다 동적으로 로딩하는 기술입니다. 메모리가 제한된 디바이스에서 대형 LLM을 실행할 수 있게 해줍니다.

**동작 원리**:
1. 모델 가중치를 플래시 저장소(파일)에 저장
2. 추론 시 필요한 가중치만 메모리에 로딩
3. `lookahead` 파라미터로 다음에 필요한 가중치를 미리 로딩 (비동기)
4. 사용이 끝난 가중치는 메모리에서 언로드

**효과**: 30B 파라미터 Qwen3-MoE 모델의 피크 메모리를 **16.5GB → 1.3GB** 로 감소.

```cpp
// NetworkGraph 생성 시 FSU 활성화
NetworkGraph graph(
  true,                    // enable_fsu
  ExecutionMode::TRAIN,    // 실행 모드
  "/path/to/weights.bin",  // fsu_path
  2,                       // lookahead (미리 로딩할 단계 수)
  "NCHW",                  // tensor_format
  "FP32-FP32"              // tensor_dtype
);
```

### 7.2 MoE Cache (Mixture of Experts)

MoE(Mixture of Experts) 모델에서 각 토큰은 전체 Expert 중 일부만 사용합니다. MoE Cache는 자주 사용되는 Expert를 메모리에 유지하고, 나머지를 플래시로 스왑하는 지능형 캐싱 메커니즘입니다.

**동작 원리**:
1. 각 Expert의 사용 빈도를 추적
2. 자주 사용되는 Expert는 메모리에 상주
3. 덜 사용되는 Expert는 플래시로 스왑 아웃
4. 필요한 Expert가 메모리에 없으면 동적으로 로딩

### 7.3 Mixed Precision Training (FP16/FP32)

가중치는 FP16으로 저장하고, 그래디언트 계산 시 FP32 마스터 가중치를 사용하는 혼합 정밀도 학습을 지원합니다.

```cpp
// NetworkGraph에서 혼합 정밀도 설정
NetworkGraph graph(..., "FP32-FP16");  // 가중치=FP32, 활성화=FP16

bool isMixedPrecision() {
  return !istrequal(tensor_dtype[1], "FP32");
}
```

**장점**:
- 메모리 사용량 감소 (활성화 텐서가 FP16)
- 계산 속도 향상 (FP16 연산)
- 수치 안정성 유지 (FP32 마스터 가중치 + Loss Scaling)

### 7.4 In-Place Execution Optimization

입력과 출력이 동일한 메모리를 공유하도록 하여 메모리 사용량을 줄이는 최적화입니다.

```cpp
enum class InPlaceType {
  NONE,           // In-Place 아님
  RESTRICTING,    // In-Place이며 앞 레이어도 In-Place여야 함
  NON_RESTRICTING // In-Place이며 앞 레이어 제약 없음
};

enum class InPlaceDirection {
  NONE, LEFT, RIGHT  // 이진 입력 시 어느 쪽 입력과 공유할지
};
```

`NetworkGraph::inPlaceOptimize()`가 컴파일 단계에서 자동으로 In-Place 가능 레이어를 탐지하고 그래프를 수정합니다.

### 7.5 LoRA Support

LoRA(Low-Rank Adaptation)는 대형 모델의 효율적인 파인튜닝을 위한 기법입니다. NNTrainer는 LoRA 어댑터를 레이어에 부착하여 전체 가중치를 업데이트하지 않고도 파인튜닝할 수 있습니다.

### 7.6 Quantization (INT4, UINT4, INT8 등)

다양한 양자화 포맷을 지원하여 모델 크기를 줄이고 추론 속도를 향상시킵니다.

| 포맷 | 설명 |
|------|------|
| **Q4_0** | GGUF 스타일 4비트 블록 양자화. 32개 원소 블록당 1개의 스케일 |
| **Q4_K** | 개선된 4비트 양자화. 슈퍼블록 구조 |
| **INT8** | 8비트 정수 양자화. Per-Tensor Affine |
| **BCQ** | Binary Code Quantization |

```cpp
// 저장 시 양자화 적용
layer->save(file, run_context, false, mode, trainable, TensorDim::DataType::Q4_0);
```

### 7.7 Platform Support

| 플랫폼 | 지원 내용 |
|--------|-----------|
| **Android RPC** | Android에서 원격 프로시저 호출을 통한 NNTrainer 실행 |
| **OpenCL** | GPU 가속을 위한 OpenCL 백엔드 (`ClContext`) |
| **Windows** | x86_64 및 ARM64 Windows 지원. libiomp (Intel OpenMP) 윈도우 빌드 포함 |
| **Tizen** | 공식 지원 플랫폼. C API 제공 |

---

## 8. 핵심 파일 레퍼런스

| 파일 경로 | 역할 | 핵심 클래스/함수 |
|-----------|------|------------------|
| `nntrainer/engine.h` | 전역 싱글톤 엔진. Context 등록, 팩토리, ThreadPool 관리 | `Engine`, `registerContext()`, `createLayerObject()` |
| `nntrainer/context.h` | 실행 컨텍스트 추상화. 하드웨어 백엔드별 환경 제공 | `Context`, `ContextData`, `ContextPluggable` |
| `nntrainer/models/neuralnet.h` | 모델 진입점. 학습/추론의 최상위 인터페이스 | `NeuralNetwork`, `compile()`, `initialize()`, `train()`, `forwarding()` |
| `nntrainer/graph/network_graph.h` | 계산 그래프 컨테이너. 레이어 관리, 위상 정렬, 메모리 관리 | `NetworkGraph`, `forwarding()`, `backwarding()`, `compile()` |
| `nntrainer/graph/graph_core.h` | 그래프 코어. 인접 리스트, 위상 정렬 | `GraphCore`, `topologicalSort()`, `makeAdjacencyList()` |
| `nntrainer/graph/graph_node.h` | 그래프 노드 추상 인터페이스 | `GraphNode`, `ExecutionOrder`, `graph_const_iterator` |
| `nntrainer/layers/layer_devel.h` | 레이어 추상 기본 클래스 | `Layer`, `forwarding()`, `calcDerivative()`, `calcGradient()` |
| `nntrainer/layers/layer_impl.h` | 가중치/편향 기반 레이어 기반 클래스 | `LayerImpl` |
| `nntrainer/layers/layer_node.h` | Layer를 감싸는 그래프 노드 래퍼 | `LayerNode`, `configureRunContext()`, `getRunContext()` |
| `nntrainer/layers/layer_context.h` | 레이어 초기화/실행 컨텍스트 | `InitLayerContext`, `RunLayerContext` |
| `nntrainer/tensor/tensor.h` | 다차원 텐서 클래스. Pimpl 패턴 | `Tensor`, `dot()`, `multiply()`, `add_i()` |
| `nntrainer/tensor/tensor_base.h` | TensorBase 추상 클래스. 데이터 타입별 구현 기반 | `TensorBase`, `FloatTensor`, `HalfTensor` |
| `nntrainer/tensor/var_grad.h` | 변수+그래디언트 쌍 관리 | `Var_Grad`, `getVariable()`, `getGradient()` |
| `nntrainer/tensor/weight.h` | Var_Grad 확장. 정규화, 옵티마이저 변수 | `Weight`, `applyGradient()`, `clipGradientByGlobalNorm()` |
| `nntrainer/tensor/manager.h` | 중앙 텐서 코디네이터. FSU, 메모리 풀 관리 | `Manager`, `requestWeights()`, `allocateTensors()` |
| `nntrainer/tensor/tensor_pool.h` | 풀드 할당기. 메모리 플래닝 | `TensorPool`, `BasicPlanner` |
| `nntrainer/tensor/memory_pool.h` | 물리적 메모리 할당 (mmap, rpcmem) | `MemoryPool`, `MMapedMemory` |
| `nntrainer/optimizers/optimizer_devel.h` | 옵티마이저 추상 클래스 | `Optimizer`, `applyGradient()` |
| `nntrainer/optimizers/optimizer_context.h` | 옵티마이저 실행 컨텍스트 | `RunOptimizerContext`, `getWeight()`, `getGradient()` |
