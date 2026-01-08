import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'
queries = [
    get_detailed_instruct(task, 'how much protein should a female eat'),
    get_detailed_instruct(task, '南瓜的家常做法')
]
# No need to add instruction for retrieval doc1ments
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
]
input_texts = queries + documents

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')

# print(model)

# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# for name, param in model.state_dict().items():
#     print(f"{name}: {param.data_ptr()}")

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
# => [[91.92852783203125, 67.580322265625], [70.3814468383789, 92.1330795288086]]

# XLMRobertaModel(
#   (embeddings): XLMRobertaEmbeddings(
#     (word_embeddings): Embedding(250002, 1024, padding_idx=1)
#     (position_embeddings): Embedding(514, 1024, padding_idx=1)
#     (token_type_embeddings): Embedding(1, 1024)
#     (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#     (dropout): Dropout(p=0.1, inplace=False)
#   )
#   (encoder): XLMRobertaEncoder(
#     (layer): ModuleList(
#       (0-23): 24 x XLMRobertaLayer(
#         (attention): XLMRobertaAttention(
#           (self): XLMRobertaSdpaSelfAttention(
#             (query): Linear(in_features=1024, out_features=1024, bias=True)
#             (key): Linear(in_features=1024, out_features=1024, bias=True)
#             (value): Linear(in_features=1024, out_features=1024, bias=True)
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#           (output): XLMRobertaSelfOutput(
#             (dense): Linear(in_features=1024, out_features=1024, bias=True)
#             (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#             (dropout): Dropout(p=0.1, inplace=False)
#           )
#         )
#         (intermediate): XLMRobertaIntermediate(
#           (dense): Linear(in_features=1024, out_features=4096, bias=True)
#           (intermediate_act_fn): GELUActivation()
#         )
#         (output): XLMRobertaOutput(
#           (dense): Linear(in_features=4096, out_features=1024, bias=True)
#           (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#           (dropout): Dropout(p=0.1, inplace=False)
#         )
#       )
#     )
#   )
#   (pooler): XLMRobertaPooler(
#     (dense): Linear(in_features=1024, out_features=1024, bias=True)
#     (activation): Tanh()
#   )
# )
# embeddings.word_embeddings.weight        torch.Size([250002, 1024])
# embeddings.position_embeddings.weight    torch.Size([514, 1024])
# embeddings.token_type_embeddings.weight          torch.Size([1, 1024])
# embeddings.LayerNorm.weight      torch.Size([1024])
# embeddings.LayerNorm.bias        torch.Size([1024])
# encoder.layer.0.attention.self.query.weight      torch.Size([1024, 1024])
# encoder.layer.0.attention.self.query.bias        torch.Size([1024])
# encoder.layer.0.attention.self.key.weight        torch.Size([1024, 1024])
# encoder.layer.0.attention.self.key.bias          torch.Size([1024])
# encoder.layer.0.attention.self.value.weight      torch.Size([1024, 1024])
# encoder.layer.0.attention.self.value.bias        torch.Size([1024])
# encoder.layer.0.attention.output.dense.weight    torch.Size([1024, 1024])
# encoder.layer.0.attention.output.dense.bias      torch.Size([1024])
# encoder.layer.0.attention.output.LayerNorm.weight        torch.Size([1024])
# encoder.layer.0.attention.output.LayerNorm.bias          torch.Size([1024])
# encoder.layer.0.intermediate.dense.weight        torch.Size([4096, 1024])
# encoder.layer.0.intermediate.dense.bias          torch.Size([4096])
# encoder.layer.0.output.dense.weight      torch.Size([1024, 4096])
# encoder.layer.0.output.dense.bias        torch.Size([1024])
# encoder.layer.0.output.LayerNorm.weight          torch.Size([1024])
# encoder.layer.0.output.LayerNorm.bias    torch.Size([1024])
# encoder.layer.1.attention.self.query.weight      torch.Size([1024, 1024])
# encoder.layer.1.attention.self.query.bias        torch.Size([1024])
# encoder.layer.1.attention.self.key.weight        torch.Size([1024, 1024])
# encoder.layer.1.attention.self.key.bias          torch.Size([1024])
# encoder.layer.1.attention.self.value.weight      torch.Size([1024, 1024])
# encoder.layer.1.attention.self.value.bias        torch.Size([1024])
# encoder.layer.1.attention.output.dense.weight    torch.Size([1024, 1024])
# encoder.layer.1.attention.output.dense.bias      torch.Size([1024])
# encoder.layer.1.attention.output.LayerNorm.weight        torch.Size([1024])
# encoder.layer.1.attention.output.LayerNorm.bias          torch.Size([1024])
# encoder.layer.1.intermediate.dense.weight        torch.Size([4096, 1024])
# encoder.layer.1.intermediate.dense.bias          torch.Size([4096])
# encoder.layer.1.output.dense.weight      torch.Size([1024, 4096])
# encoder.layer.1.output.dense.bias        torch.Size([1024])
# encoder.layer.1.output.LayerNorm.weight          torch.Size([1024])
# encoder.layer.1.output.LayerNorm.bias    torch.Size([1024])
# encoder.layer.2.attention.self.query.weight      torch.Size([1024, 1024])
# encoder.layer.2.attention.self.query.bias        torch.Size([1024])
# encoder.layer.2.attention.self.key.weight        torch.Size([1024, 1024])
# encoder.layer.2.attention.self.key.bias          torch.Size([1024])
# encoder.layer.2.attention.self.value.weight      torch.Size([1024, 1024])
# encoder.layer.2.attention.self.value.bias        torch.Size([1024])
# encoder.layer.2.attention.output.dense.weight    torch.Size([1024, 1024])
# encoder.layer.2.attention.output.dense.bias      torch.Size([1024])
# encoder.layer.2.attention.output.LayerNorm.weight        torch.Size([1024])
# encoder.layer.2.attention.output.LayerNorm.bias          torch.Size([1024])
# encoder.layer.2.intermediate.dense.weight        torch.Size([4096, 1024])
# encoder.layer.2.intermediate.dense.bias          torch.Size([4096])
# encoder.layer.2.output.dense.weight      torch.Size([1024, 4096])
# encoder.layer.2.output.dense.bias        torch.Size([1024])
# encoder.layer.2.output.LayerNorm.weight          torch.Size([1024])
# encoder.layer.2.output.LayerNorm.bias    torch.Size([1024])
# encoder.layer.3.attention.self.query.weight      torch.Size([1024, 1024])
# encoder.layer.3.attention.self.query.bias        torch.Size([1024])
# encoder.layer.3.attention.self.key.weight        torch.Size([1024, 1024])
# encoder.layer.3.attention.self.key.bias          torch.Size([1024])
# encoder.layer.3.attention.self.value.weight      torch.Size([1024, 1024])
# encoder.layer.3.attention.self.value.bias        torch.Size([1024])
# encoder.layer.3.attention.output.dense.weight    torch.Size([1024, 1024])
# encoder.layer.3.attention.output.dense.bias      torch.Size([1024])
# encoder.layer.3.attention.output.LayerNorm.weight        torch.Size([1024])
# encoder.layer.3.attention.output.LayerNorm.bias          torch.Size([1024])
# encoder.layer.3.intermediate.dense.weight        torch.Size([4096, 1024])
# encoder.layer.3.intermediate.dense.bias          torch.Size([4096])
# encoder.layer.3.output.dense.weight      torch.Size([1024, 4096])
# encoder.layer.3.output.dense.bias        torch.Size([1024])
# encoder.layer.3.output.LayerNorm.weight          torch.Size([1024])
# encoder.layer.3.output.LayerNorm.bias    torch.Size([1024])
# encoder.layer.4.attention.self.query.weight      torch.Size([1024, 1024])
# encoder.layer.4.attention.self.query.bias        torch.Size([1024])
# encoder.layer.4.attention.self.key.weight        torch.Size([1024, 1024])
# encoder.layer.4.attention.self.key.bias          torch.Size([1024])
# encoder.layer.4.attention.self.value.weight      torch.Size([1024, 1024])
# encoder.layer.4.attention.self.value.bias        torch.Size([1024])
# encoder.layer.4.attention.output.dense.weight    torch.Size([1024, 1024])
# encoder.layer.4.attention.output.dense.bias      torch.Size([1024])
# encoder.layer.4.attention.output.LayerNorm.weight        torch.Size([1024])
# encoder.layer.4.attention.output.LayerNorm.bias          torch.Size([1024])
# encoder.layer.4.intermediate.dense.weight        torch.Size([4096, 1024])
# encoder.layer.4.intermediate.dense.bias          torch.Size([4096])
# encoder.layer.4.output.dense.weight      torch.Size([1024, 4096])
# encoder.layer.4.output.dense.bias        torch.Size([1024])
# encoder.layer.4.output.LayerNorm.weight          torch.Size([1024])
# encoder.layer.4.output.LayerNorm.bias    torch.Size([1024])
# encoder.layer.5.attention.self.query.weight      torch.Size([1024, 1024])
# encoder.layer.5.attention.self.query.bias        torch.Size([1024])
# encoder.layer.5.attention.self.key.weight        torch.Size([1024, 1024])
# encoder.layer.5.attention.self.key.bias          torch.Size([1024])
# encoder.layer.5.attention.self.value.weight      torch.Size([1024, 1024])
# encoder.layer.5.attention.self.value.bias        torch.Size([1024])
# encoder.layer.5.attention.output.dense.weight    torch.Size([1024, 1024])
# encoder.layer.5.attention.output.dense.bias      torch.Size([1024])
# encoder.layer.5.attention.output.LayerNorm.weight        torch.Size([1024])
# encoder.layer.5.attention.output.LayerNorm.bias          torch.Size([1024])
# encoder.layer.5.intermediate.dense.weight        torch.Size([4096, 1024])
# encoder.layer.5.intermediate.dense.bias          torch.Size([4096])
# encoder.layer.5.output.dense.weight      torch.Size([1024, 4096])
# encoder.layer.5.output.dense.bias        torch.Size([1024])
# encoder.layer.5.output.LayerNorm.weight          torch.Size([1024])
# encoder.layer.5.output.LayerNorm.bias    torch.Size([1024])
# encoder.layer.6.attention.self.query.weight      torch.Size([1024, 1024])
# encoder.layer.6.attention.self.query.bias        torch.Size([1024])
# encoder.layer.6.attention.self.key.weight        torch.Size([1024, 1024])
# encoder.layer.6.attention.self.key.bias          torch.Size([1024])
# encoder.layer.6.attention.self.value.weight      torch.Size([1024, 1024])
# encoder.layer.6.attention.self.value.bias        torch.Size([1024])
# encoder.layer.6.attention.output.dense.weight    torch.Size([1024, 1024])
# encoder.layer.6.attention.output.dense.bias      torch.Size([1024])
# encoder.layer.6.attention.output.LayerNorm.weight        torch.Size([1024])
# encoder.layer.6.attention.output.LayerNorm.bias          torch.Size([1024])
# encoder.layer.6.intermediate.dense.weight        torch.Size([4096, 1024])
# encoder.layer.6.intermediate.dense.bias          torch.Size([4096])
# encoder.layer.6.output.dense.weight      torch.Size([1024, 4096])
# encoder.layer.6.output.dense.bias        torch.Size([1024])
# encoder.layer.6.output.LayerNorm.weight          torch.Size([1024])
# encoder.layer.6.output.LayerNorm.bias    torch.Size([1024])
# encoder.layer.7.attention.self.query.weight      torch.Size([1024, 1024])
# encoder.layer.7.attention.self.query.bias        torch.Size([1024])
# encoder.layer.7.attention.self.key.weight        torch.Size([1024, 1024])
# encoder.layer.7.attention.self.key.bias          torch.Size([1024])
# encoder.layer.7.attention.self.value.weight      torch.Size([1024, 1024])
# encoder.layer.7.attention.self.value.bias        torch.Size([1024])
# encoder.layer.7.attention.output.dense.weight    torch.Size([1024, 1024])
# encoder.layer.7.attention.output.dense.bias      torch.Size([1024])
# encoder.layer.7.attention.output.LayerNorm.weight        torch.Size([1024])
# encoder.layer.7.attention.output.LayerNorm.bias          torch.Size([1024])
# encoder.layer.7.intermediate.dense.weight        torch.Size([4096, 1024])
# encoder.layer.7.intermediate.dense.bias          torch.Size([4096])
# encoder.layer.7.output.dense.weight      torch.Size([1024, 4096])
# encoder.layer.7.output.dense.bias        torch.Size([1024])
# encoder.layer.7.output.LayerNorm.weight          torch.Size([1024])
# encoder.layer.7.output.LayerNorm.bias    torch.Size([1024])
# encoder.layer.8.attention.self.query.weight      torch.Size([1024, 1024])
# encoder.layer.8.attention.self.query.bias        torch.Size([1024])
# encoder.layer.8.attention.self.key.weight        torch.Size([1024, 1024])
# encoder.layer.8.attention.self.key.bias          torch.Size([1024])
# encoder.layer.8.attention.self.value.weight      torch.Size([1024, 1024])
# encoder.layer.8.attention.self.value.bias        torch.Size([1024])
# encoder.layer.8.attention.output.dense.weight    torch.Size([1024, 1024])
# encoder.layer.8.attention.output.dense.bias      torch.Size([1024])
# encoder.layer.8.attention.output.LayerNorm.weight        torch.Size([1024])
# encoder.layer.8.attention.output.LayerNorm.bias          torch.Size([1024])
# encoder.layer.8.intermediate.dense.weight        torch.Size([4096, 1024])
# encoder.layer.8.intermediate.dense.bias          torch.Size([4096])
# encoder.layer.8.output.dense.weight      torch.Size([1024, 4096])
# encoder.layer.8.output.dense.bias        torch.Size([1024])
# encoder.layer.8.output.LayerNorm.weight          torch.Size([1024])
# encoder.layer.8.output.LayerNorm.bias    torch.Size([1024])
# encoder.layer.9.attention.self.query.weight      torch.Size([1024, 1024])
# encoder.layer.9.attention.self.query.bias        torch.Size([1024])
# encoder.layer.9.attention.self.key.weight        torch.Size([1024, 1024])
# encoder.layer.9.attention.self.key.bias          torch.Size([1024])
# encoder.layer.9.attention.self.value.weight      torch.Size([1024, 1024])
# encoder.layer.9.attention.self.value.bias        torch.Size([1024])
# encoder.layer.9.attention.output.dense.weight    torch.Size([1024, 1024])
# encoder.layer.9.attention.output.dense.bias      torch.Size([1024])
# encoder.layer.9.attention.output.LayerNorm.weight        torch.Size([1024])
# encoder.layer.9.attention.output.LayerNorm.bias          torch.Size([1024])
# encoder.layer.9.intermediate.dense.weight        torch.Size([4096, 1024])
# encoder.layer.9.intermediate.dense.bias          torch.Size([4096])
# encoder.layer.9.output.dense.weight      torch.Size([1024, 4096])
# encoder.layer.9.output.dense.bias        torch.Size([1024])
# encoder.layer.9.output.LayerNorm.weight          torch.Size([1024])
# encoder.layer.9.output.LayerNorm.bias    torch.Size([1024])
# encoder.layer.10.attention.self.query.weight     torch.Size([1024, 1024])
# encoder.layer.10.attention.self.query.bias       torch.Size([1024])
# encoder.layer.10.attention.self.key.weight       torch.Size([1024, 1024])
# encoder.layer.10.attention.self.key.bias         torch.Size([1024])
# encoder.layer.10.attention.self.value.weight     torch.Size([1024, 1024])
# encoder.layer.10.attention.self.value.bias       torch.Size([1024])
# encoder.layer.10.attention.output.dense.weight   torch.Size([1024, 1024])
# encoder.layer.10.attention.output.dense.bias     torch.Size([1024])
# encoder.layer.10.attention.output.LayerNorm.weight       torch.Size([1024])
# encoder.layer.10.attention.output.LayerNorm.bias         torch.Size([1024])
# encoder.layer.10.intermediate.dense.weight       torch.Size([4096, 1024])
# encoder.layer.10.intermediate.dense.bias         torch.Size([4096])
# encoder.layer.10.output.dense.weight     torch.Size([1024, 4096])
# encoder.layer.10.output.dense.bias       torch.Size([1024])
# encoder.layer.10.output.LayerNorm.weight         torch.Size([1024])
# encoder.layer.10.output.LayerNorm.bias   torch.Size([1024])
# encoder.layer.11.attention.self.query.weight     torch.Size([1024, 1024])
# encoder.layer.11.attention.self.query.bias       torch.Size([1024])
# encoder.layer.11.attention.self.key.weight       torch.Size([1024, 1024])
# encoder.layer.11.attention.self.key.bias         torch.Size([1024])
# encoder.layer.11.attention.self.value.weight     torch.Size([1024, 1024])
# encoder.layer.11.attention.self.value.bias       torch.Size([1024])
# encoder.layer.11.attention.output.dense.weight   torch.Size([1024, 1024])
# encoder.layer.11.attention.output.dense.bias     torch.Size([1024])
# encoder.layer.11.attention.output.LayerNorm.weight       torch.Size([1024])
# encoder.layer.11.attention.output.LayerNorm.bias         torch.Size([1024])
# encoder.layer.11.intermediate.dense.weight       torch.Size([4096, 1024])
# encoder.layer.11.intermediate.dense.bias         torch.Size([4096])
# encoder.layer.11.output.dense.weight     torch.Size([1024, 4096])
# encoder.layer.11.output.dense.bias       torch.Size([1024])
# encoder.layer.11.output.LayerNorm.weight         torch.Size([1024])
# encoder.layer.11.output.LayerNorm.bias   torch.Size([1024])
# encoder.layer.12.attention.self.query.weight     torch.Size([1024, 1024])
# encoder.layer.12.attention.self.query.bias       torch.Size([1024])
# encoder.layer.12.attention.self.key.weight       torch.Size([1024, 1024])
# encoder.layer.12.attention.self.key.bias         torch.Size([1024])
# encoder.layer.12.attention.self.value.weight     torch.Size([1024, 1024])
# encoder.layer.12.attention.self.value.bias       torch.Size([1024])
# encoder.layer.12.attention.output.dense.weight   torch.Size([1024, 1024])
# encoder.layer.12.attention.output.dense.bias     torch.Size([1024])
# encoder.layer.12.attention.output.LayerNorm.weight       torch.Size([1024])
# encoder.layer.12.attention.output.LayerNorm.bias         torch.Size([1024])
# encoder.layer.12.intermediate.dense.weight       torch.Size([4096, 1024])
# encoder.layer.12.intermediate.dense.bias         torch.Size([4096])
# encoder.layer.12.output.dense.weight     torch.Size([1024, 4096])
# encoder.layer.12.output.dense.bias       torch.Size([1024])
# encoder.layer.12.output.LayerNorm.weight         torch.Size([1024])
# encoder.layer.12.output.LayerNorm.bias   torch.Size([1024])
# encoder.layer.13.attention.self.query.weight     torch.Size([1024, 1024])
# encoder.layer.13.attention.self.query.bias       torch.Size([1024])
# encoder.layer.13.attention.self.key.weight       torch.Size([1024, 1024])
# encoder.layer.13.attention.self.key.bias         torch.Size([1024])
# encoder.layer.13.attention.self.value.weight     torch.Size([1024, 1024])
# encoder.layer.13.attention.self.value.bias       torch.Size([1024])
# encoder.layer.13.attention.output.dense.weight   torch.Size([1024, 1024])
# encoder.layer.13.attention.output.dense.bias     torch.Size([1024])
# encoder.layer.13.attention.output.LayerNorm.weight       torch.Size([1024])
# encoder.layer.13.attention.output.LayerNorm.bias         torch.Size([1024])
# encoder.layer.13.intermediate.dense.weight       torch.Size([4096, 1024])
# encoder.layer.13.intermediate.dense.bias         torch.Size([4096])
# encoder.layer.13.output.dense.weight     torch.Size([1024, 4096])
# encoder.layer.13.output.dense.bias       torch.Size([1024])
# encoder.layer.13.output.LayerNorm.weight         torch.Size([1024])
# encoder.layer.13.output.LayerNorm.bias   torch.Size([1024])
# encoder.layer.14.attention.self.query.weight     torch.Size([1024, 1024])
# encoder.layer.14.attention.self.query.bias       torch.Size([1024])
# encoder.layer.14.attention.self.key.weight       torch.Size([1024, 1024])
# encoder.layer.14.attention.self.key.bias         torch.Size([1024])
# encoder.layer.14.attention.self.value.weight     torch.Size([1024, 1024])
# encoder.layer.14.attention.self.value.bias       torch.Size([1024])
# encoder.layer.14.attention.output.dense.weight   torch.Size([1024, 1024])
# encoder.layer.14.attention.output.dense.bias     torch.Size([1024])
# encoder.layer.14.attention.output.LayerNorm.weight       torch.Size([1024])
# encoder.layer.14.attention.output.LayerNorm.bias         torch.Size([1024])
# encoder.layer.14.intermediate.dense.weight       torch.Size([4096, 1024])
# encoder.layer.14.intermediate.dense.bias         torch.Size([4096])
# encoder.layer.14.output.dense.weight     torch.Size([1024, 4096])
# encoder.layer.14.output.dense.bias       torch.Size([1024])
# encoder.layer.14.output.LayerNorm.weight         torch.Size([1024])
# encoder.layer.14.output.LayerNorm.bias   torch.Size([1024])
# encoder.layer.15.attention.self.query.weight     torch.Size([1024, 1024])
# encoder.layer.15.attention.self.query.bias       torch.Size([1024])
# encoder.layer.15.attention.self.key.weight       torch.Size([1024, 1024])
# encoder.layer.15.attention.self.key.bias         torch.Size([1024])
# encoder.layer.15.attention.self.value.weight     torch.Size([1024, 1024])
# encoder.layer.15.attention.self.value.bias       torch.Size([1024])
# encoder.layer.15.attention.output.dense.weight   torch.Size([1024, 1024])
# encoder.layer.15.attention.output.dense.bias     torch.Size([1024])
# encoder.layer.15.attention.output.LayerNorm.weight       torch.Size([1024])
# encoder.layer.15.attention.output.LayerNorm.bias         torch.Size([1024])
# encoder.layer.15.intermediate.dense.weight       torch.Size([4096, 1024])
# encoder.layer.15.intermediate.dense.bias         torch.Size([4096])
# encoder.layer.15.output.dense.weight     torch.Size([1024, 4096])
# encoder.layer.15.output.dense.bias       torch.Size([1024])
# encoder.layer.15.output.LayerNorm.weight         torch.Size([1024])
# encoder.layer.15.output.LayerNorm.bias   torch.Size([1024])
# encoder.layer.16.attention.self.query.weight     torch.Size([1024, 1024])
# encoder.layer.16.attention.self.query.bias       torch.Size([1024])
# encoder.layer.16.attention.self.key.weight       torch.Size([1024, 1024])
# encoder.layer.16.attention.self.key.bias         torch.Size([1024])
# encoder.layer.16.attention.self.value.weight     torch.Size([1024, 1024])
# encoder.layer.16.attention.self.value.bias       torch.Size([1024])
# encoder.layer.16.attention.output.dense.weight   torch.Size([1024, 1024])
# encoder.layer.16.attention.output.dense.bias     torch.Size([1024])
# encoder.layer.16.attention.output.LayerNorm.weight       torch.Size([1024])
# encoder.layer.16.attention.output.LayerNorm.bias         torch.Size([1024])
# encoder.layer.16.intermediate.dense.weight       torch.Size([4096, 1024])
# encoder.layer.16.intermediate.dense.bias         torch.Size([4096])
# encoder.layer.16.output.dense.weight     torch.Size([1024, 4096])
# encoder.layer.16.output.dense.bias       torch.Size([1024])
# encoder.layer.16.output.LayerNorm.weight         torch.Size([1024])
# encoder.layer.16.output.LayerNorm.bias   torch.Size([1024])
# encoder.layer.17.attention.self.query.weight     torch.Size([1024, 1024])
# encoder.layer.17.attention.self.query.bias       torch.Size([1024])
# encoder.layer.17.attention.self.key.weight       torch.Size([1024, 1024])
# encoder.layer.17.attention.self.key.bias         torch.Size([1024])
# encoder.layer.17.attention.self.value.weight     torch.Size([1024, 1024])
# encoder.layer.17.attention.self.value.bias       torch.Size([1024])
# encoder.layer.17.attention.output.dense.weight   torch.Size([1024, 1024])
# encoder.layer.17.attention.output.dense.bias     torch.Size([1024])
# encoder.layer.17.attention.output.LayerNorm.weight       torch.Size([1024])
# encoder.layer.17.attention.output.LayerNorm.bias         torch.Size([1024])
# encoder.layer.17.intermediate.dense.weight       torch.Size([4096, 1024])
# encoder.layer.17.intermediate.dense.bias         torch.Size([4096])
# encoder.layer.17.output.dense.weight     torch.Size([1024, 4096])
# encoder.layer.17.output.dense.bias       torch.Size([1024])
# encoder.layer.17.output.LayerNorm.weight         torch.Size([1024])
# encoder.layer.17.output.LayerNorm.bias   torch.Size([1024])
# encoder.layer.18.attention.self.query.weight     torch.Size([1024, 1024])
# encoder.layer.18.attention.self.query.bias       torch.Size([1024])
# encoder.layer.18.attention.self.key.weight       torch.Size([1024, 1024])
# encoder.layer.18.attention.self.key.bias         torch.Size([1024])
# encoder.layer.18.attention.self.value.weight     torch.Size([1024, 1024])
# encoder.layer.18.attention.self.value.bias       torch.Size([1024])
# encoder.layer.18.attention.output.dense.weight   torch.Size([1024, 1024])
# encoder.layer.18.attention.output.dense.bias     torch.Size([1024])
# encoder.layer.18.attention.output.LayerNorm.weight       torch.Size([1024])
# encoder.layer.18.attention.output.LayerNorm.bias         torch.Size([1024])
# encoder.layer.18.intermediate.dense.weight       torch.Size([4096, 1024])
# encoder.layer.18.intermediate.dense.bias         torch.Size([4096])
# encoder.layer.18.output.dense.weight     torch.Size([1024, 4096])
# encoder.layer.18.output.dense.bias       torch.Size([1024])
# encoder.layer.18.output.LayerNorm.weight         torch.Size([1024])
# encoder.layer.18.output.LayerNorm.bias   torch.Size([1024])
# encoder.layer.19.attention.self.query.weight     torch.Size([1024, 1024])
# encoder.layer.19.attention.self.query.bias       torch.Size([1024])
# encoder.layer.19.attention.self.key.weight       torch.Size([1024, 1024])
# encoder.layer.19.attention.self.key.bias         torch.Size([1024])
# encoder.layer.19.attention.self.value.weight     torch.Size([1024, 1024])
# encoder.layer.19.attention.self.value.bias       torch.Size([1024])
# encoder.layer.19.attention.output.dense.weight   torch.Size([1024, 1024])
# encoder.layer.19.attention.output.dense.bias     torch.Size([1024])
# encoder.layer.19.attention.output.LayerNorm.weight       torch.Size([1024])
# encoder.layer.19.attention.output.LayerNorm.bias         torch.Size([1024])
# encoder.layer.19.intermediate.dense.weight       torch.Size([4096, 1024])
# encoder.layer.19.intermediate.dense.bias         torch.Size([4096])
# encoder.layer.19.output.dense.weight     torch.Size([1024, 4096])
# encoder.layer.19.output.dense.bias       torch.Size([1024])
# encoder.layer.19.output.LayerNorm.weight         torch.Size([1024])
# encoder.layer.19.output.LayerNorm.bias   torch.Size([1024])
# encoder.layer.20.attention.self.query.weight     torch.Size([1024, 1024])
# encoder.layer.20.attention.self.query.bias       torch.Size([1024])
# encoder.layer.20.attention.self.key.weight       torch.Size([1024, 1024])
# encoder.layer.20.attention.self.key.bias         torch.Size([1024])
# encoder.layer.20.attention.self.value.weight     torch.Size([1024, 1024])
# encoder.layer.20.attention.self.value.bias       torch.Size([1024])
# encoder.layer.20.attention.output.dense.weight   torch.Size([1024, 1024])
# encoder.layer.20.attention.output.dense.bias     torch.Size([1024])
# encoder.layer.20.attention.output.LayerNorm.weight       torch.Size([1024])
# encoder.layer.20.attention.output.LayerNorm.bias         torch.Size([1024])
# encoder.layer.20.intermediate.dense.weight       torch.Size([4096, 1024])
# encoder.layer.20.intermediate.dense.bias         torch.Size([4096])
# encoder.layer.20.output.dense.weight     torch.Size([1024, 4096])
# encoder.layer.20.output.dense.bias       torch.Size([1024])
# encoder.layer.20.output.LayerNorm.weight         torch.Size([1024])
# encoder.layer.20.output.LayerNorm.bias   torch.Size([1024])
# encoder.layer.21.attention.self.query.weight     torch.Size([1024, 1024])
# encoder.layer.21.attention.self.query.bias       torch.Size([1024])
# encoder.layer.21.attention.self.key.weight       torch.Size([1024, 1024])
# encoder.layer.21.attention.self.key.bias         torch.Size([1024])
# encoder.layer.21.attention.self.value.weight     torch.Size([1024, 1024])
# encoder.layer.21.attention.self.value.bias       torch.Size([1024])
# encoder.layer.21.attention.output.dense.weight   torch.Size([1024, 1024])
# encoder.layer.21.attention.output.dense.bias     torch.Size([1024])
# encoder.layer.21.attention.output.LayerNorm.weight       torch.Size([1024])
# encoder.layer.21.attention.output.LayerNorm.bias         torch.Size([1024])
# encoder.layer.21.intermediate.dense.weight       torch.Size([4096, 1024])
# encoder.layer.21.intermediate.dense.bias         torch.Size([4096])
# encoder.layer.21.output.dense.weight     torch.Size([1024, 4096])
# encoder.layer.21.output.dense.bias       torch.Size([1024])
# encoder.layer.21.output.LayerNorm.weight         torch.Size([1024])
# encoder.layer.21.output.LayerNorm.bias   torch.Size([1024])
# encoder.layer.22.attention.self.query.weight     torch.Size([1024, 1024])
# encoder.layer.22.attention.self.query.bias       torch.Size([1024])
# encoder.layer.22.attention.self.key.weight       torch.Size([1024, 1024])
# encoder.layer.22.attention.self.key.bias         torch.Size([1024])
# encoder.layer.22.attention.self.value.weight     torch.Size([1024, 1024])
# encoder.layer.22.attention.self.value.bias       torch.Size([1024])
# encoder.layer.22.attention.output.dense.weight   torch.Size([1024, 1024])
# encoder.layer.22.attention.output.dense.bias     torch.Size([1024])
# encoder.layer.22.attention.output.LayerNorm.weight       torch.Size([1024])
# encoder.layer.22.attention.output.LayerNorm.bias         torch.Size([1024])
# encoder.layer.22.intermediate.dense.weight       torch.Size([4096, 1024])
# encoder.layer.22.intermediate.dense.bias         torch.Size([4096])
# encoder.layer.22.output.dense.weight     torch.Size([1024, 4096])
# encoder.layer.22.output.dense.bias       torch.Size([1024])
# encoder.layer.22.output.LayerNorm.weight         torch.Size([1024])
# encoder.layer.22.output.LayerNorm.bias   torch.Size([1024])
# encoder.layer.23.attention.self.query.weight     torch.Size([1024, 1024])
# encoder.layer.23.attention.self.query.bias       torch.Size([1024])
# encoder.layer.23.attention.self.key.weight       torch.Size([1024, 1024])
# encoder.layer.23.attention.self.key.bias         torch.Size([1024])
# encoder.layer.23.attention.self.value.weight     torch.Size([1024, 1024])
# encoder.layer.23.attention.self.value.bias       torch.Size([1024])
# encoder.layer.23.attention.output.dense.weight   torch.Size([1024, 1024])
# encoder.layer.23.attention.output.dense.bias     torch.Size([1024])
# encoder.layer.23.attention.output.LayerNorm.weight       torch.Size([1024])
# encoder.layer.23.attention.output.LayerNorm.bias         torch.Size([1024])
# encoder.layer.23.intermediate.dense.weight       torch.Size([4096, 1024])
# encoder.layer.23.intermediate.dense.bias         torch.Size([4096])
# encoder.layer.23.output.dense.weight     torch.Size([1024, 4096])
# encoder.layer.23.output.dense.bias       torch.Size([1024])
# encoder.layer.23.output.LayerNorm.weight         torch.Size([1024])
# encoder.layer.23.output.LayerNorm.bias   torch.Size([1024])
# pooler.dense.weight      torch.Size([1024, 1024])
# pooler.dense.bias        torch.Size([1024])
# embeddings.word_embeddings.weight: 126348158771264
# embeddings.position_embeddings.weight: 237919936
# embeddings.token_type_embeddings.weight: 216338048
# embeddings.LayerNorm.weight: 234904832
# embeddings.LayerNorm.bias: 234900672
# encoder.layer.0.attention.self.query.weight: 529678144
# encoder.layer.0.attention.self.query.bias: 216358848
# encoder.layer.0.attention.self.key.weight: 244219712
# encoder.layer.0.attention.self.key.bias: 216354688
# encoder.layer.0.attention.self.value.weight: 533872512
# encoder.layer.0.attention.self.value.bias: 216274432
# encoder.layer.0.attention.output.dense.weight: 240025344
# encoder.layer.0.attention.output.dense.bias: 216350528
# encoder.layer.0.attention.output.LayerNorm.weight: 216346368
# encoder.layer.0.attention.output.LayerNorm.bias: 216342208
# encoder.layer.0.intermediate.dense.weight: 538066880
# encoder.layer.0.intermediate.dense.bias: 216278592
# encoder.layer.0.output.dense.weight: 554844160
# encoder.layer.0.output.dense.bias: 216303360
# encoder.layer.0.output.LayerNorm.weight: 216299200
# encoder.layer.0.output.LayerNorm.bias: 216295040
# encoder.layer.1.attention.self.query.weight: 580010176
# encoder.layer.1.attention.self.query.bias: 216324160
# encoder.layer.1.attention.self.key.weight: 575815808
# encoder.layer.1.attention.self.key.bias: 216320000
# encoder.layer.1.attention.self.value.weight: 584204544
# encoder.layer.1.attention.self.value.bias: 216328320
# encoder.layer.1.attention.output.dense.weight: 571621440
# encoder.layer.1.attention.output.dense.bias: 216315840
# encoder.layer.1.attention.output.LayerNorm.weight: 216311680
# encoder.layer.1.attention.output.LayerNorm.bias: 216307520
# encoder.layer.1.intermediate.dense.weight: 588398912
# encoder.layer.1.intermediate.dense.bias: 248414080
# encoder.layer.1.output.dense.weight: 605176192
# encoder.layer.1.output.dense.bias: 248438848
# encoder.layer.1.output.LayerNorm.weight: 248434688
# encoder.layer.1.output.LayerNorm.bias: 248430528
# encoder.layer.2.attention.self.query.weight: 1133662528
# encoder.layer.2.attention.self.query.bias: 248998528
# encoder.layer.2.attention.self.key.weight: 1129468160
# encoder.layer.2.attention.self.key.bias: 248994368
# encoder.layer.2.attention.self.value.weight: 1137856896
# encoder.layer.2.attention.self.value.bias: 249002688
# encoder.layer.2.attention.output.dense.weight: 1125273792
# encoder.layer.2.attention.output.dense.bias: 248990208
# encoder.layer.2.attention.output.LayerNorm.weight: 248986048
# encoder.layer.2.attention.output.LayerNorm.bias: 248981888
# encoder.layer.2.intermediate.dense.weight: 1142051264
# encoder.layer.2.intermediate.dense.bias: 249006848
# encoder.layer.2.output.dense.weight: 1158828544
# encoder.layer.2.output.dense.bias: 249031616
# encoder.layer.2.output.LayerNorm.weight: 249027456
# encoder.layer.2.output.LayerNorm.bias: 249023296
# encoder.layer.3.attention.self.query.weight: 1385322688
# encoder.layer.3.attention.self.query.bias: 249267968
# encoder.layer.3.attention.self.key.weight: 1381128320
# encoder.layer.3.attention.self.key.bias: 249263808
# encoder.layer.3.attention.self.value.weight: 1389517056
# encoder.layer.3.attention.self.value.bias: 249272128
# encoder.layer.3.attention.output.dense.weight: 1376933952
# encoder.layer.3.attention.output.dense.bias: 249259648
# encoder.layer.3.attention.output.LayerNorm.weight: 249255488
# encoder.layer.3.attention.output.LayerNorm.bias: 249251328
# encoder.layer.3.intermediate.dense.weight: 1393711424
# encoder.layer.3.intermediate.dense.bias: 249276288
# encoder.layer.3.output.dense.weight: 1410488704
# encoder.layer.3.output.dense.bias: 249301056
# encoder.layer.3.output.LayerNorm.weight: 249296896
# encoder.layer.3.output.LayerNorm.bias: 249292736
# encoder.layer.4.attention.self.query.weight: 1435654720
# encoder.layer.4.attention.self.query.bias: 249321856
# encoder.layer.4.attention.self.key.weight: 1431460352
# encoder.layer.4.attention.self.key.bias: 249317696
# encoder.layer.4.attention.self.value.weight: 1439849088
# encoder.layer.4.attention.self.value.bias: 249326016
# encoder.layer.4.attention.output.dense.weight: 1427265984
# encoder.layer.4.attention.output.dense.bias: 249313536
# encoder.layer.4.attention.output.LayerNorm.weight: 249309376
# encoder.layer.4.attention.output.LayerNorm.bias: 249305216
# encoder.layer.4.intermediate.dense.weight: 1444043456
# encoder.layer.4.intermediate.dense.bias: 249330176
# encoder.layer.4.output.dense.weight: 1460820736
# encoder.layer.4.output.dense.bias: 249354944
# encoder.layer.4.output.LayerNorm.weight: 249350784
# encoder.layer.4.output.LayerNorm.bias: 249346624
# encoder.layer.5.attention.self.query.weight: 1485986752
# encoder.layer.5.attention.self.query.bias: 249375744
# encoder.layer.5.attention.self.key.weight: 1481792384
# encoder.layer.5.attention.self.key.bias: 249371584
# encoder.layer.5.attention.self.value.weight: 1490181120
# encoder.layer.5.attention.self.value.bias: 249379904
# encoder.layer.5.attention.output.dense.weight: 1477598016
# encoder.layer.5.attention.output.dense.bias: 249367424
# encoder.layer.5.attention.output.LayerNorm.weight: 249363264
# encoder.layer.5.attention.output.LayerNorm.bias: 249359104
# encoder.layer.5.intermediate.dense.weight: 1494375488
# encoder.layer.5.intermediate.dense.bias: 249384064
# encoder.layer.5.output.dense.weight: 1511152768
# encoder.layer.5.output.dense.bias: 249408832
# encoder.layer.5.output.LayerNorm.weight: 249404672
# encoder.layer.5.output.LayerNorm.bias: 249400512
# encoder.layer.6.attention.self.query.weight: 1536318784
# encoder.layer.6.attention.self.query.bias: 249429632
# encoder.layer.6.attention.self.key.weight: 1532124416
# encoder.layer.6.attention.self.key.bias: 249425472
# encoder.layer.6.attention.self.value.weight: 1540513152
# encoder.layer.6.attention.self.value.bias: 249433792
# encoder.layer.6.attention.output.dense.weight: 1527930048
# encoder.layer.6.attention.output.dense.bias: 249421312
# encoder.layer.6.attention.output.LayerNorm.weight: 249417152
# encoder.layer.6.attention.output.LayerNorm.bias: 249412992
# encoder.layer.6.intermediate.dense.weight: 1544707520
# encoder.layer.6.intermediate.dense.bias: 249437952
# encoder.layer.6.output.dense.weight: 1561484800
# encoder.layer.6.output.dense.bias: 249462720
# encoder.layer.6.output.LayerNorm.weight: 249458560
# encoder.layer.6.output.LayerNorm.bias: 249454400
# encoder.layer.7.attention.self.query.weight: 1586650816
# encoder.layer.7.attention.self.query.bias: 249483520
# encoder.layer.7.attention.self.key.weight: 1582456448
# encoder.layer.7.attention.self.key.bias: 249479360
# encoder.layer.7.attention.self.value.weight: 1590845184
# encoder.layer.7.attention.self.value.bias: 249487680
# encoder.layer.7.attention.output.dense.weight: 1578262080
# encoder.layer.7.attention.output.dense.bias: 249475200
# encoder.layer.7.attention.output.LayerNorm.weight: 249471040
# encoder.layer.7.attention.output.LayerNorm.bias: 249466880
# encoder.layer.7.intermediate.dense.weight: 1595039552
# encoder.layer.7.intermediate.dense.bias: 249491840
# encoder.layer.7.output.dense.weight: 1611816832
# encoder.layer.7.output.dense.bias: 249516608
# encoder.layer.7.output.LayerNorm.weight: 249512448
# encoder.layer.7.output.LayerNorm.bias: 249508288
# encoder.layer.8.attention.self.query.weight: 1636982848
# encoder.layer.8.attention.self.query.bias: 249537408
# encoder.layer.8.attention.self.key.weight: 1632788480
# encoder.layer.8.attention.self.key.bias: 249533248
# encoder.layer.8.attention.self.value.weight: 1641177216
# encoder.layer.8.attention.self.value.bias: 249541568
# encoder.layer.8.attention.output.dense.weight: 1628594112
# encoder.layer.8.attention.output.dense.bias: 249529088
# encoder.layer.8.attention.output.LayerNorm.weight: 249524928
# encoder.layer.8.attention.output.LayerNorm.bias: 249520768
# encoder.layer.8.intermediate.dense.weight: 1645371584
# encoder.layer.8.intermediate.dense.bias: 249545728
# encoder.layer.8.output.dense.weight: 1662148864
# encoder.layer.8.output.dense.bias: 249570496
# encoder.layer.8.output.LayerNorm.weight: 249566336
# encoder.layer.8.output.LayerNorm.bias: 249562176
# encoder.layer.9.attention.self.query.weight: 1687314880
# encoder.layer.9.attention.self.query.bias: 249591296
# encoder.layer.9.attention.self.key.weight: 1683120512
# encoder.layer.9.attention.self.key.bias: 249587136
# encoder.layer.9.attention.self.value.weight: 1691509248
# encoder.layer.9.attention.self.value.bias: 249595456
# encoder.layer.9.attention.output.dense.weight: 1678926144
# encoder.layer.9.attention.output.dense.bias: 249582976
# encoder.layer.9.attention.output.LayerNorm.weight: 249578816
# encoder.layer.9.attention.output.LayerNorm.bias: 249574656
# encoder.layer.9.intermediate.dense.weight: 1695703616
# encoder.layer.9.intermediate.dense.bias: 249599616
# encoder.layer.9.output.dense.weight: 1712480896
# encoder.layer.9.output.dense.bias: 249624384
# encoder.layer.9.output.LayerNorm.weight: 249620224
# encoder.layer.9.output.LayerNorm.bias: 249616064
# encoder.layer.10.attention.self.query.weight: 630342208
# encoder.layer.10.attention.self.query.bias: 248459648
# encoder.layer.10.attention.self.key.weight: 626147840
# encoder.layer.10.attention.self.key.bias: 248455488
# encoder.layer.10.attention.self.value.weight: 634536576
# encoder.layer.10.attention.self.value.bias: 248463808
# encoder.layer.10.attention.output.dense.weight: 621953472
# encoder.layer.10.attention.output.dense.bias: 248451328
# encoder.layer.10.attention.output.LayerNorm.weight: 248447168
# encoder.layer.10.attention.output.LayerNorm.bias: 248443008
# encoder.layer.10.intermediate.dense.weight: 638730944
# encoder.layer.10.intermediate.dense.bias: 248467968
# encoder.layer.10.output.dense.weight: 655508224
# encoder.layer.10.output.dense.bias: 248492736
# encoder.layer.10.output.LayerNorm.weight: 248488576
# encoder.layer.10.output.LayerNorm.bias: 248484416
# encoder.layer.11.attention.self.query.weight: 680674240
# encoder.layer.11.attention.self.query.bias: 248513536
# encoder.layer.11.attention.self.key.weight: 676479872
# encoder.layer.11.attention.self.key.bias: 248509376
# encoder.layer.11.attention.self.value.weight: 684868608
# encoder.layer.11.attention.self.value.bias: 248517696
# encoder.layer.11.attention.output.dense.weight: 672285504
# encoder.layer.11.attention.output.dense.bias: 248505216
# encoder.layer.11.attention.output.LayerNorm.weight: 248501056
# encoder.layer.11.attention.output.LayerNorm.bias: 248496896
# encoder.layer.11.intermediate.dense.weight: 689062976
# encoder.layer.11.intermediate.dense.bias: 248521856
# encoder.layer.11.output.dense.weight: 705840256
# encoder.layer.11.output.dense.bias: 248546624
# encoder.layer.11.output.LayerNorm.weight: 248542464
# encoder.layer.11.output.LayerNorm.bias: 248538304
# encoder.layer.12.attention.self.query.weight: 731006272
# encoder.layer.12.attention.self.query.bias: 248567424
# encoder.layer.12.attention.self.key.weight: 726811904
# encoder.layer.12.attention.self.key.bias: 248563264
# encoder.layer.12.attention.self.value.weight: 735200640
# encoder.layer.12.attention.self.value.bias: 248571584
# encoder.layer.12.attention.output.dense.weight: 722617536
# encoder.layer.12.attention.output.dense.bias: 248559104
# encoder.layer.12.attention.output.LayerNorm.weight: 248554944
# encoder.layer.12.attention.output.LayerNorm.bias: 248550784
# encoder.layer.12.intermediate.dense.weight: 739395008
# encoder.layer.12.intermediate.dense.bias: 248575744
# encoder.layer.12.output.dense.weight: 756172288
# encoder.layer.12.output.dense.bias: 248600512
# encoder.layer.12.output.LayerNorm.weight: 248596352
# encoder.layer.12.output.LayerNorm.bias: 248592192
# encoder.layer.13.attention.self.query.weight: 781338304
# encoder.layer.13.attention.self.query.bias: 248621312
# encoder.layer.13.attention.self.key.weight: 777143936
# encoder.layer.13.attention.self.key.bias: 248617152
# encoder.layer.13.attention.self.value.weight: 785532672
# encoder.layer.13.attention.self.value.bias: 248625472
# encoder.layer.13.attention.output.dense.weight: 772949568
# encoder.layer.13.attention.output.dense.bias: 248612992
# encoder.layer.13.attention.output.LayerNorm.weight: 248608832
# encoder.layer.13.attention.output.LayerNorm.bias: 248604672
# encoder.layer.13.intermediate.dense.weight: 789727040
# encoder.layer.13.intermediate.dense.bias: 248629632
# encoder.layer.13.output.dense.weight: 806504320
# encoder.layer.13.output.dense.bias: 248654400
# encoder.layer.13.output.LayerNorm.weight: 248650240
# encoder.layer.13.output.LayerNorm.bias: 248646080
# encoder.layer.14.attention.self.query.weight: 831670336
# encoder.layer.14.attention.self.query.bias: 248675200
# encoder.layer.14.attention.self.key.weight: 827475968
# encoder.layer.14.attention.self.key.bias: 248671040
# encoder.layer.14.attention.self.value.weight: 835864704
# encoder.layer.14.attention.self.value.bias: 248679360
# encoder.layer.14.attention.output.dense.weight: 823281600
# encoder.layer.14.attention.output.dense.bias: 248666880
# encoder.layer.14.attention.output.LayerNorm.weight: 248662720
# encoder.layer.14.attention.output.LayerNorm.bias: 248658560
# encoder.layer.14.intermediate.dense.weight: 840059072
# encoder.layer.14.intermediate.dense.bias: 248683520
# encoder.layer.14.output.dense.weight: 856836352
# encoder.layer.14.output.dense.bias: 248708288
# encoder.layer.14.output.LayerNorm.weight: 248704128
# encoder.layer.14.output.LayerNorm.bias: 248699968
# encoder.layer.15.attention.self.query.weight: 882002368
# encoder.layer.15.attention.self.query.bias: 248729088
# encoder.layer.15.attention.self.key.weight: 877808000
# encoder.layer.15.attention.self.key.bias: 248724928
# encoder.layer.15.attention.self.value.weight: 886196736
# encoder.layer.15.attention.self.value.bias: 248733248
# encoder.layer.15.attention.output.dense.weight: 873613632
# encoder.layer.15.attention.output.dense.bias: 248720768
# encoder.layer.15.attention.output.LayerNorm.weight: 248716608
# encoder.layer.15.attention.output.LayerNorm.bias: 248712448
# encoder.layer.15.intermediate.dense.weight: 890391104
# encoder.layer.15.intermediate.dense.bias: 248737408
# encoder.layer.15.output.dense.weight: 907168384
# encoder.layer.15.output.dense.bias: 248762176
# encoder.layer.15.output.LayerNorm.weight: 248758016
# encoder.layer.15.output.LayerNorm.bias: 248753856
# encoder.layer.16.attention.self.query.weight: 932334400
# encoder.layer.16.attention.self.query.bias: 248782976
# encoder.layer.16.attention.self.key.weight: 928140032
# encoder.layer.16.attention.self.key.bias: 248778816
# encoder.layer.16.attention.self.value.weight: 936528768
# encoder.layer.16.attention.self.value.bias: 248787136
# encoder.layer.16.attention.output.dense.weight: 923945664
# encoder.layer.16.attention.output.dense.bias: 248774656
# encoder.layer.16.attention.output.LayerNorm.weight: 248770496
# encoder.layer.16.attention.output.LayerNorm.bias: 248766336
# encoder.layer.16.intermediate.dense.weight: 940723136
# encoder.layer.16.intermediate.dense.bias: 248791296
# encoder.layer.16.output.dense.weight: 957500416
# encoder.layer.16.output.dense.bias: 248816064
# encoder.layer.16.output.LayerNorm.weight: 248811904
# encoder.layer.16.output.LayerNorm.bias: 248807744
# encoder.layer.17.attention.self.query.weight: 982666432
# encoder.layer.17.attention.self.query.bias: 248836864
# encoder.layer.17.attention.self.key.weight: 978472064
# encoder.layer.17.attention.self.key.bias: 248832704
# encoder.layer.17.attention.self.value.weight: 986860800
# encoder.layer.17.attention.self.value.bias: 248841024
# encoder.layer.17.attention.output.dense.weight: 974277696
# encoder.layer.17.attention.output.dense.bias: 248828544
# encoder.layer.17.attention.output.LayerNorm.weight: 248824384
# encoder.layer.17.attention.output.LayerNorm.bias: 248820224
# encoder.layer.17.intermediate.dense.weight: 991055168
# encoder.layer.17.intermediate.dense.bias: 248845184
# encoder.layer.17.output.dense.weight: 1007832448
# encoder.layer.17.output.dense.bias: 248869952
# encoder.layer.17.output.LayerNorm.weight: 248865792
# encoder.layer.17.output.LayerNorm.bias: 248861632
# encoder.layer.18.attention.self.query.weight: 1032998464
# encoder.layer.18.attention.self.query.bias: 248890752
# encoder.layer.18.attention.self.key.weight: 1028804096
# encoder.layer.18.attention.self.key.bias: 248886592
# encoder.layer.18.attention.self.value.weight: 1037192832
# encoder.layer.18.attention.self.value.bias: 248894912
# encoder.layer.18.attention.output.dense.weight: 1024609728
# encoder.layer.18.attention.output.dense.bias: 248882432
# encoder.layer.18.attention.output.LayerNorm.weight: 248878272
# encoder.layer.18.attention.output.LayerNorm.bias: 248874112
# encoder.layer.18.intermediate.dense.weight: 1041387200
# encoder.layer.18.intermediate.dense.bias: 248899072
# encoder.layer.18.output.dense.weight: 1058164480
# encoder.layer.18.output.dense.bias: 248923840
# encoder.layer.18.output.LayerNorm.weight: 248919680
# encoder.layer.18.output.LayerNorm.bias: 248915520
# encoder.layer.19.attention.self.query.weight: 1083330496
# encoder.layer.19.attention.self.query.bias: 248944640
# encoder.layer.19.attention.self.key.weight: 1079136128
# encoder.layer.19.attention.self.key.bias: 248940480
# encoder.layer.19.attention.self.value.weight: 1087524864
# encoder.layer.19.attention.self.value.bias: 248948800
# encoder.layer.19.attention.output.dense.weight: 1074941760
# encoder.layer.19.attention.output.dense.bias: 248936320
# encoder.layer.19.attention.output.LayerNorm.weight: 248932160
# encoder.layer.19.attention.output.LayerNorm.bias: 248928000
# encoder.layer.19.intermediate.dense.weight: 1091719232
# encoder.layer.19.intermediate.dense.bias: 248952960
# encoder.layer.19.output.dense.weight: 1108496512
# encoder.layer.19.output.dense.bias: 248977728
# encoder.layer.19.output.LayerNorm.weight: 248973568
# encoder.layer.19.output.LayerNorm.bias: 248969408
# encoder.layer.20.attention.self.query.weight: 1183994560
# encoder.layer.20.attention.self.query.bias: 249052416
# encoder.layer.20.attention.self.key.weight: 1179800192
# encoder.layer.20.attention.self.key.bias: 249048256
# encoder.layer.20.attention.self.value.weight: 1188188928
# encoder.layer.20.attention.self.value.bias: 249056576
# encoder.layer.20.attention.output.dense.weight: 1175605824
# encoder.layer.20.attention.output.dense.bias: 249044096
# encoder.layer.20.attention.output.LayerNorm.weight: 249039936
# encoder.layer.20.attention.output.LayerNorm.bias: 249035776
# encoder.layer.20.intermediate.dense.weight: 1192383296
# encoder.layer.20.intermediate.dense.bias: 249060736
# encoder.layer.20.output.dense.weight: 1209160576
# encoder.layer.20.output.dense.bias: 249085504
# encoder.layer.20.output.LayerNorm.weight: 249081344
# encoder.layer.20.output.LayerNorm.bias: 249077184
# encoder.layer.21.attention.self.query.weight: 1234326592
# encoder.layer.21.attention.self.query.bias: 249106304
# encoder.layer.21.attention.self.key.weight: 1230132224
# encoder.layer.21.attention.self.key.bias: 249102144
# encoder.layer.21.attention.self.value.weight: 1238520960
# encoder.layer.21.attention.self.value.bias: 249110464
# encoder.layer.21.attention.output.dense.weight: 1225937856
# encoder.layer.21.attention.output.dense.bias: 249097984
# encoder.layer.21.attention.output.LayerNorm.weight: 249093824
# encoder.layer.21.attention.output.LayerNorm.bias: 249089664
# encoder.layer.21.intermediate.dense.weight: 1242715328
# encoder.layer.21.intermediate.dense.bias: 249114624
# encoder.layer.21.output.dense.weight: 1259492608
# encoder.layer.21.output.dense.bias: 249139392
# encoder.layer.21.output.LayerNorm.weight: 249135232
# encoder.layer.21.output.LayerNorm.bias: 249131072
# encoder.layer.22.attention.self.query.weight: 1284658624
# encoder.layer.22.attention.self.query.bias: 249160192
# encoder.layer.22.attention.self.key.weight: 1280464256
# encoder.layer.22.attention.self.key.bias: 249156032
# encoder.layer.22.attention.self.value.weight: 1288852992
# encoder.layer.22.attention.self.value.bias: 249164352
# encoder.layer.22.attention.output.dense.weight: 1276269888
# encoder.layer.22.attention.output.dense.bias: 249151872
# encoder.layer.22.attention.output.LayerNorm.weight: 249147712
# encoder.layer.22.attention.output.LayerNorm.bias: 249143552
# encoder.layer.22.intermediate.dense.weight: 1293047360
# encoder.layer.22.intermediate.dense.bias: 249168512
# encoder.layer.22.output.dense.weight: 1309824640
# encoder.layer.22.output.dense.bias: 249193280
# encoder.layer.22.output.LayerNorm.weight: 249189120
# encoder.layer.22.output.LayerNorm.bias: 249184960
# encoder.layer.23.attention.self.query.weight: 1334990656
# encoder.layer.23.attention.self.query.bias: 249214080
# encoder.layer.23.attention.self.key.weight: 1330796288
# encoder.layer.23.attention.self.key.bias: 249209920
# encoder.layer.23.attention.self.value.weight: 1339185024
# encoder.layer.23.attention.self.value.bias: 249218240
# encoder.layer.23.attention.output.dense.weight: 1326601920
# encoder.layer.23.attention.output.dense.bias: 249205760
# encoder.layer.23.attention.output.LayerNorm.weight: 249201600
# encoder.layer.23.attention.output.LayerNorm.bias: 249197440
# encoder.layer.23.intermediate.dense.weight: 1343379392
# encoder.layer.23.intermediate.dense.bias: 249222400
# encoder.layer.23.output.dense.weight: 1360156672
# encoder.layer.23.output.dense.bias: 249247168
# encoder.layer.23.output.LayerNorm.weight: 249243008
# encoder.layer.23.output.LayerNorm.bias: 249238848
# pooler.dense.weight: 1729258176
# pooler.dense.bias: 249628544
