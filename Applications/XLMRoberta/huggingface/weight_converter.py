## @file weight_converter.py
## @brief weight conversion script for XLMRoberta model
## @author Eunju Yang <ej.yang@samsung.com>

import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel

total_size = 0
def save_xlmroberta_for_nntrainer(params, n_layers, dtype, file):
    """Convert and save weights as nntrainer format for XLMRoberta model"""
    
    def save_weight(weight):
        np.array(weight, dtype=dtype).tofile(file)  

    # Save embedding layer weights in the correct order
    save_weight(params["embeddings.word_embeddings.weight"])
    save_weight(params["embeddings.token_type_embeddings.weight"])
    save_weight(params["embeddings.position_embeddings.weight"])
    save_weight(params["embeddings.LayerNorm.weight"])
    save_weight(params["embeddings.LayerNorm.bias"])

    # Process all layers
    for layer_idx in range(n_layers):
        layer_prefix = f"encoder.layer.{layer_idx}."

        # Save attention layer weights
        # Self attention weights (Q, K, V)
        save_weight(params[f"{layer_prefix}attention.self.query.weight"].permute(1, 0))
        save_weight(params[f"{layer_prefix}attention.self.query.bias"])
        save_weight(params[f"{layer_prefix}attention.self.key.weight"].permute(1, 0))
        save_weight(params[f"{layer_prefix}attention.self.key.bias"])
        save_weight(params[f"{layer_prefix}attention.self.value.weight"].permute(1, 0))
        save_weight(params[f"{layer_prefix}attention.self.value.bias"])
        
        # Attention output weights
        save_weight(params[f"{layer_prefix}attention.output.dense.weight"].permute(1, 0))
        save_weight(params[f"{layer_prefix}attention.output.dense.bias"])
        save_weight(params[f"{layer_prefix}attention.output.LayerNorm.weight"])
        save_weight(params[f"{layer_prefix}attention.output.LayerNorm.bias"])

        # Save feed forward layer weights
        save_weight(params[f"{layer_prefix}intermediate.dense.weight"].permute(1, 0))
        save_weight(params[f"{layer_prefix}intermediate.dense.bias"])
        save_weight(params[f"{layer_prefix}output.dense.weight"].permute(1, 0))
        save_weight(params[f"{layer_prefix}output.dense.bias"])
        save_weight(params[f"{layer_prefix}output.LayerNorm.weight"])
        save_weight(params[f"{layer_prefix}output.LayerNorm.bias"])

    # Save pooler layer weights
    save_weight(params["pooler.dense.weight"].permute(1, 0))
    save_weight(params["pooler.dense.bias"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./multilingual-e5-large-instruct")
    parser.add_argument("--output_name", type=str, default="./nntr_xlmroberta.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    args = parser.parse_args()
    
    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, dtype="float", trust_remote_code=True)
    model.eval()

    for param_tensor in model.state_dict():
     weight = model.state_dict()[param_tensor]
     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
     print(weight)
     print("----------------------------------------------")

    with open(output_name, "wb") as f_model :
        save_xlmroberta_for_nntrainer(model.state_dict(), config.num_hidden_layers, data_dtype, f_model)
