import torch.nn.functional as F

from torch import Tensor

from transformers import AutoTokenizer, AutoModel
# from sentence_transformers import SentenceTransformer

import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

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
print(queries)
print(documents)

input_texts = queries + documents
# input_texts = "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
# input_texts = "hello world"
# input_texts = ["My name is James", 'hello', 'good night', 'I love you']
print(input_texts)

# tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
tokenizer = AutoTokenizer.from_pretrained("/home/junbong/src/AI/nntrainer/Applications/XLMRoberta/huggingface/multilingual-e5-large-instruct")
model = AutoModel.from_pretrained("/home/junbong/src/AI/nntrainer/Applications/XLMRoberta/huggingface/multilingual-e5-large-instruct")

# print(model)
for param_tensor in model.state_dict():
     weight = model.state_dict()[param_tensor]
     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
     print(weight)     
     print("----------------------------------------------")

# # for name, param in model.state_dict().items():
#     print(f"{name}: {param.data_ptr()}")

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
print("Printing Result:")
outputs = model(**batch_dict)
print("pooler_output: ")
print(outputs.pooler_output.shape)
print(outputs.pooler_output)
print("----------------------------------------------")

embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
# => [[91.92852783203125, 67.580322265625], [70.3814468383789, 92.1330795288086]]

# XLMRobertaModel(
#   (embeddings): XLMRobertaEmbeddings(
#     (word_embeddings): Embedding(250002, 1024, padding_idx=1)
#     (token_type_embeddings): Embedding(1, 1024)
#     (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
#     (dropout): Dropout(p=0.1, inplace=False)
#     (position_embeddings): Embedding(514, 1024, padding_idx=1)
#   )
#   (encoder): XLMRobertaEncoder(
#     (layer): ModuleList(
#       (0-23): 24 x XLMRobertaLayer(
#         (attention): XLMRobertaAttention(
#           (self): XLMRobertaSelfAttention(
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
# embeddings.token_type_embeddings.weight          torch.Size([1, 1024])
# embeddings.LayerNorm.weight      torch.Size([1024])
# embeddings.LayerNorm.bias        torch.Size([1024])
# embeddings.position_embeddings.weight    torch.Size([514, 1024])
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
# embeddings.word_embeddings.weight: 128361177851664
# embeddings.token_type_embeddings.weight: 128361177849616
# embeddings.LayerNorm.weight: 128361176794896
# embeddings.LayerNorm.bias: 128361176792848
# embeddings.position_embeddings.weight: 128361176796944
# encoder.layer.0.attention.self.query.weight: 128361694060304
# encoder.layer.0.attention.self.query.bias: 128361694058256
# encoder.layer.0.attention.self.key.weight: 128361691961104
# encoder.layer.0.attention.self.key.bias: 128361691959056
# encoder.layer.0.attention.self.value.weight: 128361696159504
# encoder.layer.0.attention.self.value.bias: 128361696157456
# encoder.layer.0.attention.output.dense.weight: 128361689861904
# encoder.layer.0.attention.output.dense.bias: 128361689859856
# encoder.layer.0.attention.output.LayerNorm.weight: 128361689857808
# encoder.layer.0.attention.output.LayerNorm.bias: 128361689855760
# encoder.layer.0.intermediate.dense.weight: 128361698264848
# encoder.layer.0.intermediate.dense.bias: 128361698256656
# encoder.layer.0.output.dense.weight: 128361706659600
# encoder.layer.0.output.dense.bias: 128361706657552
# encoder.layer.0.output.LayerNorm.weight: 128361706655504
# encoder.layer.0.output.LayerNorm.bias: 128361706653456
# encoder.layer.1.attention.self.query.weight: 128361719252752
# encoder.layer.1.attention.self.query.bias: 128361719250704
# encoder.layer.1.attention.self.key.weight: 128361717153552
# encoder.layer.1.attention.self.key.bias: 128361717151504
# encoder.layer.1.attention.self.value.weight: 128361721351952
# encoder.layer.1.attention.self.value.bias: 128361721349904
# encoder.layer.1.attention.output.dense.weight: 128361715054352
# encoder.layer.1.attention.output.dense.bias: 128361715052304
# encoder.layer.1.attention.output.LayerNorm.weight: 128361715050256
# encoder.layer.1.attention.output.LayerNorm.bias: 128361715048208
# encoder.layer.1.intermediate.dense.weight: 128361723457296
# encoder.layer.1.intermediate.dense.bias: 128361723449104
# encoder.layer.1.output.dense.weight: 128361731852048
# encoder.layer.1.output.dense.bias: 128361731850000
# encoder.layer.1.output.LayerNorm.weight: 128361731847952
# encoder.layer.1.output.LayerNorm.bias: 128361731845904
# encoder.layer.2.attention.self.query.weight: 128361996369680
# encoder.layer.2.attention.self.query.bias: 128361996367632
# encoder.layer.2.attention.self.key.weight: 128361994270480
# encoder.layer.2.attention.self.key.bias: 128361994268432
# encoder.layer.2.attention.self.value.weight: 128361998468880
# encoder.layer.2.attention.self.value.bias: 128361998466832
# encoder.layer.2.attention.output.dense.weight: 128361992171280
# encoder.layer.2.attention.output.dense.bias: 128361992169232
# encoder.layer.2.attention.output.LayerNorm.weight: 128361992167184
# encoder.layer.2.attention.output.LayerNorm.bias: 128361992165136
# encoder.layer.2.intermediate.dense.weight: 128362000574224
# encoder.layer.2.intermediate.dense.bias: 128362000566032
# encoder.layer.2.output.dense.weight: 128362008968976
# encoder.layer.2.output.dense.bias: 128362008966928
# encoder.layer.2.output.LayerNorm.weight: 128362008964880
# encoder.layer.2.output.LayerNorm.bias: 128362008962832
# encoder.layer.3.attention.self.query.weight: 128362122331920
# encoder.layer.3.attention.self.query.bias: 128362122329872
# encoder.layer.3.attention.self.key.weight: 128362120232720
# encoder.layer.3.attention.self.key.bias: 128362120230672
# encoder.layer.3.attention.self.value.weight: 128362124431120
# encoder.layer.3.attention.self.value.bias: 128362124429072
# encoder.layer.3.attention.output.dense.weight: 128362118133520
# encoder.layer.3.attention.output.dense.bias: 128362118131472
# encoder.layer.3.attention.output.LayerNorm.weight: 128362118129424
# encoder.layer.3.attention.output.LayerNorm.bias: 128362118127376
# encoder.layer.3.intermediate.dense.weight: 128362126536464
# encoder.layer.3.intermediate.dense.bias: 128362126528272
# encoder.layer.3.output.dense.weight: 128362134931216
# encoder.layer.3.output.dense.bias: 128362134929168
# encoder.layer.3.output.LayerNorm.weight: 128362134927120
# encoder.layer.3.output.LayerNorm.bias: 128362134925072
# encoder.layer.4.attention.self.query.weight: 128362147524368
# encoder.layer.4.attention.self.query.bias: 128362147522320
# encoder.layer.4.attention.self.key.weight: 128362145425168
# encoder.layer.4.attention.self.key.bias: 128362145423120
# encoder.layer.4.attention.self.value.weight: 128362149623568
# encoder.layer.4.attention.self.value.bias: 128362149621520
# encoder.layer.4.attention.output.dense.weight: 128362143325968
# encoder.layer.4.attention.output.dense.bias: 128362143323920
# encoder.layer.4.attention.output.LayerNorm.weight: 128362143321872
# encoder.layer.4.attention.output.LayerNorm.bias: 128362143319824
# encoder.layer.4.intermediate.dense.weight: 128362151728912
# encoder.layer.4.intermediate.dense.bias: 128362151720720
# encoder.layer.4.output.dense.weight: 128362160123664
# encoder.layer.4.output.dense.bias: 128362160121616
# encoder.layer.4.output.LayerNorm.weight: 128362160119568
# encoder.layer.4.output.LayerNorm.bias: 128362160117520
# encoder.layer.5.attention.self.query.weight: 128362172716816
# encoder.layer.5.attention.self.query.bias: 128362172714768
# encoder.layer.5.attention.self.key.weight: 128362170617616
# encoder.layer.5.attention.self.key.bias: 128362170615568
# encoder.layer.5.attention.self.value.weight: 128362174816016
# encoder.layer.5.attention.self.value.bias: 128362174813968
# encoder.layer.5.attention.output.dense.weight: 128362168518416
# encoder.layer.5.attention.output.dense.bias: 128362168516368
# encoder.layer.5.attention.output.LayerNorm.weight: 128362168514320
# encoder.layer.5.attention.output.LayerNorm.bias: 128362168512272
# encoder.layer.5.intermediate.dense.weight: 128362176921360
# encoder.layer.5.intermediate.dense.bias: 128362176913168
# encoder.layer.5.output.dense.weight: 128362185316112
# encoder.layer.5.output.dense.bias: 128362185314064
# encoder.layer.5.output.LayerNorm.weight: 128362185312016
# encoder.layer.5.output.LayerNorm.bias: 128362185309968
# encoder.layer.6.attention.self.query.weight: 128362197909264
# encoder.layer.6.attention.self.query.bias: 128362197907216
# encoder.layer.6.attention.self.key.weight: 128362195810064
# encoder.layer.6.attention.self.key.bias: 128362195808016
# encoder.layer.6.attention.self.value.weight: 128362200008464
# encoder.layer.6.attention.self.value.bias: 128362200006416
# encoder.layer.6.attention.output.dense.weight: 128362193710864
# encoder.layer.6.attention.output.dense.bias: 128362193708816
# encoder.layer.6.attention.output.LayerNorm.weight: 128362193706768
# encoder.layer.6.attention.output.LayerNorm.bias: 128362193704720
# encoder.layer.6.intermediate.dense.weight: 128362202113808
# encoder.layer.6.intermediate.dense.bias: 128362202105616
# encoder.layer.6.output.dense.weight: 128362210508560
# encoder.layer.6.output.dense.bias: 128362210506512
# encoder.layer.6.output.LayerNorm.weight: 128362210504464
# encoder.layer.6.output.LayerNorm.bias: 128362210502416
# encoder.layer.7.attention.self.query.weight: 128362223101712
# encoder.layer.7.attention.self.query.bias: 128362223099664
# encoder.layer.7.attention.self.key.weight: 128362221002512
# encoder.layer.7.attention.self.key.bias: 128362221000464
# encoder.layer.7.attention.self.value.weight: 128362225200912
# encoder.layer.7.attention.self.value.bias: 128362225198864
# encoder.layer.7.attention.output.dense.weight: 128362218903312
# encoder.layer.7.attention.output.dense.bias: 128362218901264
# encoder.layer.7.attention.output.LayerNorm.weight: 128362218899216
# encoder.layer.7.attention.output.LayerNorm.bias: 128362218897168
# encoder.layer.7.intermediate.dense.weight: 128362227306256
# encoder.layer.7.intermediate.dense.bias: 128362227298064
# encoder.layer.7.output.dense.weight: 128362235701008
# encoder.layer.7.output.dense.bias: 128362235698960
# encoder.layer.7.output.LayerNorm.weight: 128362235696912
# encoder.layer.7.output.LayerNorm.bias: 128362235694864
# encoder.layer.8.attention.self.query.weight: 128362248294160
# encoder.layer.8.attention.self.query.bias: 128362248292112
# encoder.layer.8.attention.self.key.weight: 128362246194960
# encoder.layer.8.attention.self.key.bias: 128362246192912
# encoder.layer.8.attention.self.value.weight: 128362250393360
# encoder.layer.8.attention.self.value.bias: 128362250391312
# encoder.layer.8.attention.output.dense.weight: 128362244095760
# encoder.layer.8.attention.output.dense.bias: 128362244093712
# encoder.layer.8.attention.output.LayerNorm.weight: 128362244091664
# encoder.layer.8.attention.output.LayerNorm.bias: 128362244089616
# encoder.layer.8.intermediate.dense.weight: 128362252498704
# encoder.layer.8.intermediate.dense.bias: 128362252490512
# encoder.layer.8.output.dense.weight: 128362260893456
# encoder.layer.8.output.dense.bias: 128362260891408
# encoder.layer.8.output.LayerNorm.weight: 128362260889360
# encoder.layer.8.output.LayerNorm.bias: 128362260887312
# encoder.layer.9.attention.self.query.weight: 128362273486608
# encoder.layer.9.attention.self.query.bias: 128362273484560
# encoder.layer.9.attention.self.key.weight: 128362271387408
# encoder.layer.9.attention.self.key.bias: 128362271385360
# encoder.layer.9.attention.self.value.weight: 128362275585808
# encoder.layer.9.attention.self.value.bias: 128362275583760
# encoder.layer.9.attention.output.dense.weight: 128362269288208
# encoder.layer.9.attention.output.dense.bias: 128362269286160
# encoder.layer.9.attention.output.LayerNorm.weight: 128362269284112
# encoder.layer.9.attention.output.LayerNorm.bias: 128362269282064
# encoder.layer.9.intermediate.dense.weight: 128362277691152
# encoder.layer.9.intermediate.dense.bias: 128362277682960
# encoder.layer.9.output.dense.weight: 128362286085904
# encoder.layer.9.output.dense.bias: 128362286083856
# encoder.layer.9.output.LayerNorm.weight: 128362286081808
# encoder.layer.9.output.LayerNorm.bias: 128362286079760
# encoder.layer.10.attention.self.query.weight: 128361744445200
# encoder.layer.10.attention.self.query.bias: 128361744443152
# encoder.layer.10.attention.self.key.weight: 128361742346000
# encoder.layer.10.attention.self.key.bias: 128361742343952
# encoder.layer.10.attention.self.value.weight: 128361746544400
# encoder.layer.10.attention.self.value.bias: 128361746542352
# encoder.layer.10.attention.output.dense.weight: 128361740246800
# encoder.layer.10.attention.output.dense.bias: 128361740244752
# encoder.layer.10.attention.output.LayerNorm.weight: 128361740242704
# encoder.layer.10.attention.output.LayerNorm.bias: 128361740240656
# encoder.layer.10.intermediate.dense.weight: 128361748649744
# encoder.layer.10.intermediate.dense.bias: 128361748641552
# encoder.layer.10.output.dense.weight: 128361757044496
# encoder.layer.10.output.dense.bias: 128361757042448
# encoder.layer.10.output.LayerNorm.weight: 128361757040400
# encoder.layer.10.output.LayerNorm.bias: 128361757038352
# encoder.layer.11.attention.self.query.weight: 128361769637648
# encoder.layer.11.attention.self.query.bias: 128361769635600
# encoder.layer.11.attention.self.key.weight: 128361767538448
# encoder.layer.11.attention.self.key.bias: 128361767536400
# encoder.layer.11.attention.self.value.weight: 128361771736848
# encoder.layer.11.attention.self.value.bias: 128361771734800
# encoder.layer.11.attention.output.dense.weight: 128361765439248
# encoder.layer.11.attention.output.dense.bias: 128361765437200
# encoder.layer.11.attention.output.LayerNorm.weight: 128361765435152
# encoder.layer.11.attention.output.LayerNorm.bias: 128361765433104
# encoder.layer.11.intermediate.dense.weight: 128361773842192
# encoder.layer.11.intermediate.dense.bias: 128361773834000
# encoder.layer.11.output.dense.weight: 128361782236944
# encoder.layer.11.output.dense.bias: 128361782234896
# encoder.layer.11.output.LayerNorm.weight: 128361782232848
# encoder.layer.11.output.LayerNorm.bias: 128361782230800
# encoder.layer.12.attention.self.query.weight: 128361794830096
# encoder.layer.12.attention.self.query.bias: 128361794828048
# encoder.layer.12.attention.self.key.weight: 128361792730896
# encoder.layer.12.attention.self.key.bias: 128361792728848
# encoder.layer.12.attention.self.value.weight: 128361796929296
# encoder.layer.12.attention.self.value.bias: 128361796927248
# encoder.layer.12.attention.output.dense.weight: 128361790631696
# encoder.layer.12.attention.output.dense.bias: 128361790629648
# encoder.layer.12.attention.output.LayerNorm.weight: 128361790627600
# encoder.layer.12.attention.output.LayerNorm.bias: 128361790625552
# encoder.layer.12.intermediate.dense.weight: 128361799034640
# encoder.layer.12.intermediate.dense.bias: 128361799026448
# encoder.layer.12.output.dense.weight: 128361807429392
# encoder.layer.12.output.dense.bias: 128361807427344
# encoder.layer.12.output.LayerNorm.weight: 128361807425296
# encoder.layer.12.output.LayerNorm.bias: 128361807423248
# encoder.layer.13.attention.self.query.weight: 128361820022544
# encoder.layer.13.attention.self.query.bias: 128361820020496
# encoder.layer.13.attention.self.key.weight: 128361817923344
# encoder.layer.13.attention.self.key.bias: 128361817921296
# encoder.layer.13.attention.self.value.weight: 128361822121744
# encoder.layer.13.attention.self.value.bias: 128361822119696
# encoder.layer.13.attention.output.dense.weight: 128361815824144
# encoder.layer.13.attention.output.dense.bias: 128361815822096
# encoder.layer.13.attention.output.LayerNorm.weight: 128361815820048
# encoder.layer.13.attention.output.LayerNorm.bias: 128361815818000
# encoder.layer.13.intermediate.dense.weight: 128361824227088
# encoder.layer.13.intermediate.dense.bias: 128361824218896
# encoder.layer.13.output.dense.weight: 128361832621840
# encoder.layer.13.output.dense.bias: 128361832619792
# encoder.layer.13.output.LayerNorm.weight: 128361832617744
# encoder.layer.13.output.LayerNorm.bias: 128361832615696
# encoder.layer.14.attention.self.query.weight: 128361845214992
# encoder.layer.14.attention.self.query.bias: 128361845212944
# encoder.layer.14.attention.self.key.weight: 128361843115792
# encoder.layer.14.attention.self.key.bias: 128361843113744
# encoder.layer.14.attention.self.value.weight: 128361847314192
# encoder.layer.14.attention.self.value.bias: 128361847312144
# encoder.layer.14.attention.output.dense.weight: 128361841016592
# encoder.layer.14.attention.output.dense.bias: 128361841014544
# encoder.layer.14.attention.output.LayerNorm.weight: 128361841012496
# encoder.layer.14.attention.output.LayerNorm.bias: 128361841010448
# encoder.layer.14.intermediate.dense.weight: 128361849419536
# encoder.layer.14.intermediate.dense.bias: 128361849411344
# encoder.layer.14.output.dense.weight: 128361857814288
# encoder.layer.14.output.dense.bias: 128361857812240
# encoder.layer.14.output.LayerNorm.weight: 128361857810192
# encoder.layer.14.output.LayerNorm.bias: 128361857808144
# encoder.layer.15.attention.self.query.weight: 128361870407440
# encoder.layer.15.attention.self.query.bias: 128361870405392
# encoder.layer.15.attention.self.key.weight: 128361868308240
# encoder.layer.15.attention.self.key.bias: 128361868306192
# encoder.layer.15.attention.self.value.weight: 128361872506640
# encoder.layer.15.attention.self.value.bias: 128361872504592
# encoder.layer.15.attention.output.dense.weight: 128361866209040
# encoder.layer.15.attention.output.dense.bias: 128361866206992
# encoder.layer.15.attention.output.LayerNorm.weight: 128361866204944
# encoder.layer.15.attention.output.LayerNorm.bias: 128361866202896
# encoder.layer.15.intermediate.dense.weight: 128361874611984
# encoder.layer.15.intermediate.dense.bias: 128361874603792
# encoder.layer.15.output.dense.weight: 128361883006736
# encoder.layer.15.output.dense.bias: 128361883004688
# encoder.layer.15.output.LayerNorm.weight: 128361883002640
# encoder.layer.15.output.LayerNorm.bias: 128361883000592
# encoder.layer.16.attention.self.query.weight: 128361895599888
# encoder.layer.16.attention.self.query.bias: 128361895597840
# encoder.layer.16.attention.self.key.weight: 128361893500688
# encoder.layer.16.attention.self.key.bias: 128361893498640
# encoder.layer.16.attention.self.value.weight: 128361897699088
# encoder.layer.16.attention.self.value.bias: 128361897697040
# encoder.layer.16.attention.output.dense.weight: 128361891401488
# encoder.layer.16.attention.output.dense.bias: 128361891399440
# encoder.layer.16.attention.output.LayerNorm.weight: 128361891397392
# encoder.layer.16.attention.output.LayerNorm.bias: 128361891395344
# encoder.layer.16.intermediate.dense.weight: 128361899804432
# encoder.layer.16.intermediate.dense.bias: 128361899796240
# encoder.layer.16.output.dense.weight: 128361908199184
# encoder.layer.16.output.dense.bias: 128361908197136
# encoder.layer.16.output.LayerNorm.weight: 128361908195088
# encoder.layer.16.output.LayerNorm.bias: 128361908193040
# encoder.layer.17.attention.self.query.weight: 128361920792336
# encoder.layer.17.attention.self.query.bias: 128361920790288
# encoder.layer.17.attention.self.key.weight: 128361918693136
# encoder.layer.17.attention.self.key.bias: 128361918691088
# encoder.layer.17.attention.self.value.weight: 128361922891536
# encoder.layer.17.attention.self.value.bias: 128361922889488
# encoder.layer.17.attention.output.dense.weight: 128361916593936
# encoder.layer.17.attention.output.dense.bias: 128361916591888
# encoder.layer.17.attention.output.LayerNorm.weight: 128361916589840
# encoder.layer.17.attention.output.LayerNorm.bias: 128361916587792
# encoder.layer.17.intermediate.dense.weight: 128361924996880
# encoder.layer.17.intermediate.dense.bias: 128361924988688
# encoder.layer.17.output.dense.weight: 128361933391632
# encoder.layer.17.output.dense.bias: 128361933389584
# encoder.layer.17.output.LayerNorm.weight: 128361933387536
# encoder.layer.17.output.LayerNorm.bias: 128361933385488
# encoder.layer.18.attention.self.query.weight: 128361945984784
# encoder.layer.18.attention.self.query.bias: 128361945982736
# encoder.layer.18.attention.self.key.weight: 128361943885584
# encoder.layer.18.attention.self.key.bias: 128361943883536
# encoder.layer.18.attention.self.value.weight: 128361948083984
# encoder.layer.18.attention.self.value.bias: 128361948081936
# encoder.layer.18.attention.output.dense.weight: 128361941786384
# encoder.layer.18.attention.output.dense.bias: 128361941784336
# encoder.layer.18.attention.output.LayerNorm.weight: 128361941782288
# encoder.layer.18.attention.output.LayerNorm.bias: 128361941780240
# encoder.layer.18.intermediate.dense.weight: 128361950189328
# encoder.layer.18.intermediate.dense.bias: 128361950181136
# encoder.layer.18.output.dense.weight: 128361958584080
# encoder.layer.18.output.dense.bias: 128361958582032
# encoder.layer.18.output.LayerNorm.weight: 128361958579984
# encoder.layer.18.output.LayerNorm.bias: 128361958577936
# encoder.layer.19.attention.self.query.weight: 128361971177232
# encoder.layer.19.attention.self.query.bias: 128361971175184
# encoder.layer.19.attention.self.key.weight: 128361969078032
# encoder.layer.19.attention.self.key.bias: 128361969075984
# encoder.layer.19.attention.self.value.weight: 128361973276432
# encoder.layer.19.attention.self.value.bias: 128361973274384
# encoder.layer.19.attention.output.dense.weight: 128361966978832
# encoder.layer.19.attention.output.dense.bias: 128361966976784
# encoder.layer.19.attention.output.LayerNorm.weight: 128361966974736
# encoder.layer.19.attention.output.LayerNorm.bias: 128361966972688
# encoder.layer.19.intermediate.dense.weight: 128361975381776
# encoder.layer.19.intermediate.dense.bias: 128361975373584
# encoder.layer.19.output.dense.weight: 128361983776528
# encoder.layer.19.output.dense.bias: 128361983774480
# encoder.layer.19.output.LayerNorm.weight: 128361983772432
# encoder.layer.19.output.LayerNorm.bias: 128361983770384
# encoder.layer.20.attention.self.query.weight: 128362021562128
# encoder.layer.20.attention.self.query.bias: 128362021560080
# encoder.layer.20.attention.self.key.weight: 128362019462928
# encoder.layer.20.attention.self.key.bias: 128362019460880
# encoder.layer.20.attention.self.value.weight: 128362023661328
# encoder.layer.20.attention.self.value.bias: 128362023659280
# encoder.layer.20.attention.output.dense.weight: 128362017363728
# encoder.layer.20.attention.output.dense.bias: 128362017361680
# encoder.layer.20.attention.output.LayerNorm.weight: 128362017359632
# encoder.layer.20.attention.output.LayerNorm.bias: 128362017357584
# encoder.layer.20.intermediate.dense.weight: 128362025766672
# encoder.layer.20.intermediate.dense.bias: 128362025758480
# encoder.layer.20.output.dense.weight: 128362034161424
# encoder.layer.20.output.dense.bias: 128362034159376
# encoder.layer.20.output.LayerNorm.weight: 128362034157328
# encoder.layer.20.output.LayerNorm.bias: 128362034155280
# encoder.layer.21.attention.self.query.weight: 128362046754576
# encoder.layer.21.attention.self.query.bias: 128362046752528
# encoder.layer.21.attention.self.key.weight: 128362044655376
# encoder.layer.21.attention.self.key.bias: 128362044653328
# encoder.layer.21.attention.self.value.weight: 128362048853776
# encoder.layer.21.attention.self.value.bias: 128362048851728
# encoder.layer.21.attention.output.dense.weight: 128362042556176
# encoder.layer.21.attention.output.dense.bias: 128362042554128
# encoder.layer.21.attention.output.LayerNorm.weight: 128362042552080
# encoder.layer.21.attention.output.LayerNorm.bias: 128362042550032
# encoder.layer.21.intermediate.dense.weight: 128362050959120
# encoder.layer.21.intermediate.dense.bias: 128362050950928
# encoder.layer.21.output.dense.weight: 128362059353872
# encoder.layer.21.output.dense.bias: 128362059351824
# encoder.layer.21.output.LayerNorm.weight: 128362059349776
# encoder.layer.21.output.LayerNorm.bias: 128362059347728
# encoder.layer.22.attention.self.query.weight: 128362071947024
# encoder.layer.22.attention.self.query.bias: 128362071944976
# encoder.layer.22.attention.self.key.weight: 128362069847824
# encoder.layer.22.attention.self.key.bias: 128362069845776
# encoder.layer.22.attention.self.value.weight: 128362074046224
# encoder.layer.22.attention.self.value.bias: 128362074044176
# encoder.layer.22.attention.output.dense.weight: 128362067748624
# encoder.layer.22.attention.output.dense.bias: 128362067746576
# encoder.layer.22.attention.output.LayerNorm.weight: 128362067744528
# encoder.layer.22.attention.output.LayerNorm.bias: 128362067742480
# encoder.layer.22.intermediate.dense.weight: 128362076151568
# encoder.layer.22.intermediate.dense.bias: 128362076143376
# encoder.layer.22.output.dense.weight: 128362084546320
# encoder.layer.22.output.dense.bias: 128362084544272
# encoder.layer.22.output.LayerNorm.weight: 128362084542224
# encoder.layer.22.output.LayerNorm.bias: 128362084540176
# encoder.layer.23.attention.self.query.weight: 128362097139472
# encoder.layer.23.attention.self.query.bias: 128362097137424
# encoder.layer.23.attention.self.key.weight: 128362095040272
# encoder.layer.23.attention.self.key.bias: 128362095038224
# encoder.layer.23.attention.self.value.weight: 128362099238672
# encoder.layer.23.attention.self.value.bias: 128362099236624
# encoder.layer.23.attention.output.dense.weight: 128362092941072
# encoder.layer.23.attention.output.dense.bias: 128362092939024
# encoder.layer.23.attention.output.LayerNorm.weight: 128362092936976
# encoder.layer.23.attention.output.LayerNorm.bias: 128362092934928
# encoder.layer.23.intermediate.dense.weight: 128362101344016
# encoder.layer.23.intermediate.dense.bias: 128362101335824
# encoder.layer.23.output.dense.weight: 128362109738768
# encoder.layer.23.output.dense.bias: 128362109736720
# encoder.layer.23.output.LayerNorm.weight: 128362109734672
# encoder.layer.23.output.LayerNorm.bias: 128362109732624
# pooler.dense.weight: 128362294476560
# pooler.dense.bias: 128362294474512

