# PR #3902: CausalLM Migrate to `ml::train::Tensor` Symbolic Graph API

- **Author**: jijoongmoon
- **Date**: 2026-04-27
- **Branch**: `feature/causallm-tensor-api` → `main`
- **Size**: +11,238 / -2,819 lines (100 files)
- **Status**: OPEN (Wait 4 #3901)
- **Depends on**: #3896 (Tensor API), #3901 (QNN 통합을 통한 compute_ops)

---

## 왜 필요한가?

CausalLM(LLM 추론용) 코드가 아직 이전 방식인 `addLayer()` + 문자열 `input_layers`로 모델을 빌드하고 있었다. PR #3896에서 도입된 새로운 Symbolic Tensor Graph API는 다른 샘플 앱(MNIST, ResNet, PicoGPT, LLaMA, YOLO 등)에는 적용되었지만, CausalLM과 그 파생 모델들(Qwen2/3, Gemma3, GPT-OSS 등)은 아직 마이그레이션되지 않았다.

이 PR은 CausalLM을 포함한 **모든 Transformer 기반 모델**을 Symbolic Tensor Graph API로 전환한다. 이는 곧이어 진행될 KV Cache 외장화 트랙(#3903-3904)의 기반이 된다.

---

## 어떤 변화가 있는가?

### 1. Transformer 기본 클래스 시그니처 변경

**이전 — 문자열 기반, `std::vector<LayerHandle>` 반환:**
```cpp
// transformer.h (이전)
class Transformer {
  virtual std::vector<LayerHandle> createTransformerDecoderBlock(
      int layer_id, std::string input_name);
  
  virtual std::vector<LayerHandle> createAttention(
      int layer_id, int seq_len, int n_heads, int head_dim,
      std::string query_name, std::string key_name, std::string value_name);
  
  virtual std::vector<LayerHandle> createMlp(
      int layer_id, int dim, int hidden_dim, std::string input_name);
  
  void constructModel();  // void 반환, model->addLayer()로 직접 추가
};
```

**이후 — Tensor 기반, `Tensor` 반환:**
```cpp
// transformer.h (이후)
class Transformer {
  virtual Tensor createTransformerDecoderBlock(int layer_id, Tensor input);
  
  virtual Tensor createAttention(int layer_id, int seq_len, int n_heads,
      int head_dim, Tensor query, Tensor key, Tensor value);
  
  virtual Tensor createMlp(int layer_id, int dim, int hidden_dim,
      Tensor input);
  
  std::pair<Tensor, Tensor> constructModel();  // {input, output} 쌍 반환
};
```

**핵심 차이:**
- 각 메서드가 `std::vector<LayerHandle>`(추가할 레이어 목록) 대신 **`Tensor`(다음 단계의 입력)**를 반환
- 레이어 간 연결이 문자열 이름 대신 **함수형 체이닝**으로 이루어짐

### 2. `constructModel()` 패턴 변화

**이전:**
```cpp
void Transformer::constructModel() {
  std::vector<LayerHandle> layers;
  layers.push_back(createLayer("input", ...));
  layers.push_back(createLayer("embedding", ...));
  
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block = createTransformerDecoderBlock(i, "layer" + ...);
    layers.insert(layers.end(), block.begin(), block.end());
  }
  
  for (auto &layer : layers) {
    model->addLayer(layer);  // 하나씩 추가
  }
  model->compile();
  model->initialize();
}
```

**이후:**
```cpp
std::pair<Tensor, Tensor> Transformer::constructModel() {
  Tensor x({1, 1, 1, INIT_SEQ_LEN}, "input0");
  
  LayerHandle embedding(createLayer("embedding", ...));
  Tensor h = embedding(x);
  
  for (int i = 0; i < NUM_LAYERS; i++) {
    h = createTransformerDecoderBlock(i, h);  // 함수형 체이닝
  }
  
  LayerHandle out_norm(createLayer("rms_norm", ...));
  h = out_norm(h);
  
  return {x, h};  // {입력, 출력} 반환 → compile(x, y, INFERENCE)에서 사용
}

// initialize()에서:
auto [x, y] = constructModel();
model->compile(x, y, ml::train::ExecutionMode::INFERENCE);
// compile(x, y)가 내부적으로 compile() + initialize() + allocate()를 호출
```

### 3. 다중 입력 연산 — Vector Form

```cpp
// mha_core (Q, K, V 3개 입력)
LayerHandle mha(createLayer("mha_core", {...}));
Tensor attn = mha({q, k, v});              // initializer_list

// residual connection (2개 입력 덧셈)
LayerHandle add(createLayer("addition", {...}));
Tensor residual = add({input, attn_out});

// swiglu (gate, up 2개 입력)
LayerHandle swiglu(createLayer("swiglu", {...}));
Tensor act = swiglu({gate, up});

// GeGLU (multiply gate*up)
LayerHandle mul(createLayer("multiply", {...}));
Tensor geglu = mul({gate_gelu, up});
```

`LayerHandle::operator()`가 `std::initializer_list<Tensor>`와 `std::vector<Tensor>`를 모두 지원한다.

### 4. 마이그레이션된 모델들

| 모델 | 특이사항 |
|------|----------|
| **Qwen2** | `createAttention` — Q, K, V 각각에 fully_connected → mha_core |
| **Qwen3** | `createAttention` — Q와 K에 reshaped RMS norm 적용 후 attention |
| **Qwen3 MoE** | `createMlp` — qwen_moe 레이어 사용 |
| **Qwen3 Slim MoE** | `createMlp` — moe_slim 레이어 사용 |
| **Qwen3 Cached Slim MoE** | `createMlp` — moe_cached_slim 레이어 사용 |
| **Gemma3** | `createAttention` — q/k norm + `createMlp` — GeGLU (gate * up) + 별도 post_attention_norm, post_ffn_norm |
| **GPT-OSS** | `createAttention` — sink + yarn rope-scaling + `createMlp` — gpt_oss_moe |
| **GPT-OSS Cached Slim** | `createMlp` — gpt_oss_moe_slim_cached |

MoE 변형 모델들은 기본 클래스(Qwen3Transformer)의 `createAttention`을 상속받아 **자동으로 마이그레이션**된다.

### 5. SentenceTransformer 적응

```cpp
// 이전: addModule이 void 반환, model->addLayer() 직접 호출
void SentenceTransformer::addModule(const std::string &type, int idx) {
  LayerHandle layer(createLayer(...));
  model->addLayer(layer);
}

// 이후: addModule이 Tensor 입력 → Tensor 출력으로 체이닝
Tensor SentenceTransformer::addModule(const std::string &type, int idx,
                                      Tensor input) {
  LayerHandle layer(createLayer(...));
  return layer(input);  // 함수형 체이닝
}

// 지원되지 않는 모듈 타입은 입력을 그대로 통과 → 체인 유지
```

---

## 기존 코드와의 비교

| 측면 | 이전 | 이후 |
|------|------|------|
| 레이어 타입 | `std::shared_ptr<ml::train::Layer>` | `ml::train::LayerHandle` |
| 텐서 타입 | 없음 (문자열 이름) | `ml::train::Tensor` |
| `createAttention` 반환 | `std::vector<LayerHandle>` | `Tensor` |
| `constructModel()` 반환 | `void` | `std::pair<Tensor, Tensor>` |
| 모델 컴파일 | `model->compile()` + `model->initialize()` (2단계) | `model->compile(x, y, INFERENCE)` (1단계) |
| MoE 상속 | 각 MoE 변형이 `createAttention` 재정의 필요 | 기본 클래스 상속으로 자동 적용 |

---

## 관련 PR

- **#3896**: Tensor API + Symbolic Graph (이 PR이 의존하는 API)
- **#3903**: KVCacheManager (이 PR의 `constructModel()` 반환 패턴에 의존)
- **#3904**: MHA Core 5-input external cache (이 PR의 `createAttention` Tensor 시그니처에 의존)