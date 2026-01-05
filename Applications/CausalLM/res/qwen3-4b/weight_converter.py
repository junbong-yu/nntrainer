## @file weight_converter.py
## @brief weight conversion script for qwen3 model
## @author Eunju Yang <ej.yang@samsung.com>

import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

total_size = 0
def save_qwen3_for_nntrainer(params, n_layers, dtype, file):  
    """Convert and save weights as nntrainer format for multi-head attention model"""  
      
    def save_weight(weight):
        np.array(weight, dtype=dtype).tofile(file)  

    def save_projection(layer_name, proj_name):  
        """Helper function to handle base/lora weight saving"""  
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"  
        if lora_key in params:  
            save_weight(params[f"{layer_name}{proj_name}.base_layer.weight"].permute(1, 0))  
            save_weight(params[f"{layer_name}{proj_name}.lora_A.default.weight"].permute(1, 0))  
            save_weight(params[f"{layer_name}{proj_name}.lora_B.default.weight"].permute(1, 0))  
        else:  
            save_weight(params[f"{layer_name}{proj_name}.weight"].permute(1, 0))  

    def save_attention(layer_name):  
        """Save attention layer weights"""  
        save_weight(params[f"{layer_name}input_layernorm.weight"])  
          
        # Save Q/K/V/O projections using helper  
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:  
            save_projection(layer_name, f"self_attn.{proj}")  
            # Qwen3
            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                print(proj_norm_name)
                save_weight(params[proj_norm_name])

    def save_feed_forward(layer_name):  
        """Save feed forward layer weights"""  
        save_weight(params[f"{layer_name}post_attention_layernorm.weight"])  
          
        # Save MLP projections using helper  
        for proj in ["up_proj", "gate_proj", "down_proj"]:  
            save_projection(layer_name, f"mlp.{proj}")  

    # Save embedding layer  
    save_weight(params["model.embed_tokens.weight"])  

    # Process all layers  
    for layer_idx in range(n_layers):  
        layer_prefix = f"model.layers.{layer_idx}."  
        save_attention(layer_prefix)  
        save_feed_forward(layer_prefix)  

    # Save final layers  
    save_weight(params["model.norm.weight"])  
    save_weight(params["lm_head.weight"].permute(1, 0))  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--output_name", type=str, default="./nntr_qwen3_4b_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    args = parser.parse_args()
    
    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="float", trust_remote_code=True)
    model.eval()

    print(model)

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    for name, param in model.state_dict().items():
        print(f"{name}: {param.data_ptr()}")

#    with open(output_name, "wb") as f_model :
#        save_qwen3_for_nntrainer(model.state_dict(), config.num_hidden_layers, data_dtype, f_model)
####################################################################################################
# Qwen3ForCausalLM(
#   (model): Qwen3Model(
#     (embed_tokens): Embedding(151936, 2560)
#     (layers): ModuleList(
#       (0-35): 36 x Qwen3DecoderLayer(
#         (self_attn): Qwen3Attention(
#           (q_proj): Linear(in_features=2560, out_features=4096, bias=False)
#           (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
#           (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
#           (o_proj): Linear(in_features=4096, out_features=2560, bias=False)
#           (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
#           (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
#         )
#         (mlp): Qwen3MLP(
#           (gate_proj): Linear(in_features=2560, out_features=9728, bias=False)
#           (up_proj): Linear(in_features=2560, out_features=9728, bias=False)
#           (down_proj): Linear(in_features=9728, out_features=2560, bias=False)
#           (act_fn): SiLUActivation()
#         )
#         (input_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
#         (post_attention_layernorm): Qwen3RMSNorm((2560,), eps=1e-06)
#       )
#     )
#     (norm): Qwen3RMSNorm((2560,), eps=1e-06)
#     (rotary_emb): Qwen3RotaryEmbedding()
#   )
#   (lm_head): Linear(in_features=2560, out_features=151936, bias=False)
# )
# model.embed_tokens.weight        torch.Size([151936, 2560])
# model.layers.0.self_attn.q_proj.weight   torch.Size([4096, 2560])
# model.layers.0.self_attn.k_proj.weight   torch.Size([1024, 2560])
# model.layers.0.self_attn.v_proj.weight   torch.Size([1024, 2560])
# model.layers.0.self_attn.o_proj.weight   torch.Size([2560, 4096])
# model.layers.0.self_attn.q_norm.weight   torch.Size([128])
# model.layers.0.self_attn.k_norm.weight   torch.Size([128])
# model.layers.0.mlp.gate_proj.weight      torch.Size([9728, 2560])
# model.layers.0.mlp.up_proj.weight        torch.Size([9728, 2560])
# model.layers.0.mlp.down_proj.weight      torch.Size([2560, 9728])
# model.layers.0.input_layernorm.weight    torch.Size([2560])
# model.layers.0.post_attention_layernorm.weight   torch.Size([2560])
# model.layers.1.self_attn.q_proj.weight   torch.Size([4096, 2560])
# model.layers.1.self_attn.k_proj.weight   torch.Size([1024, 2560])
# model.layers.1.self_attn.v_proj.weight   torch.Size([1024, 2560])
# model.layers.1.self_attn.o_proj.weight   torch.Size([2560, 4096])
# model.layers.1.self_attn.q_norm.weight   torch.Size([128])
# model.layers.1.self_attn.k_norm.weight   torch.Size([128])
# model.layers.1.mlp.gate_proj.weight      torch.Size([9728, 2560])
# model.layers.1.mlp.up_proj.weight        torch.Size([9728, 2560])
# model.layers.1.mlp.down_proj.weight      torch.Size([2560, 9728])
# model.layers.1.input_layernorm.weight    torch.Size([2560])
# model.layers.1.post_attention_layernorm.weight   torch.Size([2560])
# model.layers.2.self_attn.q_proj.weight   torch.Size([4096, 2560])
# model.layers.2.self_attn.k_proj.weight   torch.Size([1024, 2560])
# model.layers.2.self_attn.v_proj.weight   torch.Size([1024, 2560])
# model.layers.2.self_attn.o_proj.weight   torch.Size([2560, 4096])
# model.layers.2.self_attn.q_norm.weight   torch.Size([128])
# model.layers.2.self_attn.k_norm.weight   torch.Size([128])
# model.layers.2.mlp.gate_proj.weight      torch.Size([9728, 2560])
# model.layers.2.mlp.up_proj.weight        torch.Size([9728, 2560])
# model.layers.2.mlp.down_proj.weight      torch.Size([2560, 9728])
# model.layers.2.input_layernorm.weight    torch.Size([2560])
# model.layers.2.post_attention_layernorm.weight   torch.Size([2560])
# model.layers.3.self_attn.q_proj.weight   torch.Size([4096, 2560])
# model.layers.3.self_attn.k_proj.weight   torch.Size([1024, 2560])
# model.layers.3.self_attn.v_proj.weight   torch.Size([1024, 2560])
# model.layers.3.self_attn.o_proj.weight   torch.Size([2560, 4096])
# model.layers.3.self_attn.q_norm.weight   torch.Size([128])
# model.layers.3.self_attn.k_norm.weight   torch.Size([128])
# model.layers.3.mlp.gate_proj.weight      torch.Size([9728, 2560])
# model.layers.3.mlp.up_proj.weight        torch.Size([9728, 2560])
# model.layers.3.mlp.down_proj.weight      torch.Size([2560, 9728])
# model.layers.3.input_layernorm.weight    torch.Size([2560])
# model.layers.3.post_attention_layernorm.weight   torch.Size([2560])
# model.layers.4.self_attn.q_proj.weight   torch.Size([4096, 2560])
# model.layers.4.self_attn.k_proj.weight   torch.Size([1024, 2560])
# model.layers.4.self_attn.v_proj.weight   torch.Size([1024, 2560])
# model.layers.4.self_attn.o_proj.weight   torch.Size([2560, 4096])
# model.layers.4.self_attn.q_norm.weight   torch.Size([128])
# model.layers.4.self_attn.k_norm.weight   torch.Size([128])
# model.layers.4.mlp.gate_proj.weight      torch.Size([9728, 2560])
# model.layers.4.mlp.up_proj.weight        torch.Size([9728, 2560])
# model.layers.4.mlp.down_proj.weight      torch.Size([2560, 9728])
# model.layers.4.input_layernorm.weight    torch.Size([2560])
# model.layers.4.post_attention_layernorm.weight   torch.Size([2560])
# model.layers.5.self_attn.q_proj.weight   torch.Size([4096, 2560])
# model.layers.5.self_attn.k_proj.weight   torch.Size([1024, 2560])
# model.layers.5.self_attn.v_proj.weight   torch.Size([1024, 2560])
# model.layers.5.self_attn.o_proj.weight   torch.Size([2560, 4096])
# model.layers.5.self_attn.q_norm.weight   torch.Size([128])
# model.layers.5.self_attn.k_norm.weight   torch.Size([128])
# model.layers.5.mlp.gate_proj.weight      torch.Size([9728, 2560])
# model.layers.5.mlp.up_proj.weight        torch.Size([9728, 2560])
# model.layers.5.mlp.down_proj.weight      torch.Size([2560, 9728])
# model.layers.5.input_layernorm.weight    torch.Size([2560])
# model.layers.5.post_attention_layernorm.weight   torch.Size([2560])
# model.layers.6.self_attn.q_proj.weight   torch.Size([4096, 2560])
# model.layers.6.self_attn.k_proj.weight   torch.Size([1024, 2560])
# model.layers.6.self_attn.v_proj.weight   torch.Size([1024, 2560])
# model.layers.6.self_attn.o_proj.weight   torch.Size([2560, 4096])
# model.layers.6.self_attn.q_norm.weight   torch.Size([128])
# model.layers.6.self_attn.k_norm.weight   torch.Size([128])
# model.layers.6.mlp.gate_proj.weight      torch.Size([9728, 2560])
# model.layers.6.mlp.up_proj.weight        torch.Size([9728, 2560])
# model.layers.6.mlp.down_proj.weight      torch.Size([2560, 9728])
# model.layers.6.input_layernorm.weight    torch.Size([2560])
# model.layers.6.post_attention_layernorm.weight   torch.Size([2560])
# model.layers.7.self_attn.q_proj.weight   torch.Size([4096, 2560])
# model.layers.7.self_attn.k_proj.weight   torch.Size([1024, 2560])
# model.layers.7.self_attn.v_proj.weight   torch.Size([1024, 2560])
# model.layers.7.self_attn.o_proj.weight   torch.Size([2560, 4096])
# model.layers.7.self_attn.q_norm.weight   torch.Size([128])
# model.layers.7.self_attn.k_norm.weight   torch.Size([128])
# model.layers.7.mlp.gate_proj.weight      torch.Size([9728, 2560])
# model.layers.7.mlp.up_proj.weight        torch.Size([9728, 2560])
# model.layers.7.mlp.down_proj.weight      torch.Size([2560, 9728])
# model.layers.7.input_layernorm.weight    torch.Size([2560])
# model.layers.7.post_attention_layernorm.weight   torch.Size([2560])
# model.layers.8.self_attn.q_proj.weight   torch.Size([4096, 2560])
# model.layers.8.self_attn.k_proj.weight   torch.Size([1024, 2560])
# model.layers.8.self_attn.v_proj.weight   torch.Size([1024, 2560])
# model.layers.8.self_attn.o_proj.weight   torch.Size([2560, 4096])
# model.layers.8.self_attn.q_norm.weight   torch.Size([128])
# model.layers.8.self_attn.k_norm.weight   torch.Size([128])
# model.layers.8.mlp.gate_proj.weight      torch.Size([9728, 2560])
# model.layers.8.mlp.up_proj.weight        torch.Size([9728, 2560])
# model.layers.8.mlp.down_proj.weight      torch.Size([2560, 9728])
# model.layers.8.input_layernorm.weight    torch.Size([2560])
# model.layers.8.post_attention_layernorm.weight   torch.Size([2560])
# model.layers.9.self_attn.q_proj.weight   torch.Size([4096, 2560])
# model.layers.9.self_attn.k_proj.weight   torch.Size([1024, 2560])
# model.layers.9.self_attn.v_proj.weight   torch.Size([1024, 2560])
# model.layers.9.self_attn.o_proj.weight   torch.Size([2560, 4096])
# model.layers.9.self_attn.q_norm.weight   torch.Size([128])
# model.layers.9.self_attn.k_norm.weight   torch.Size([128])
# model.layers.9.mlp.gate_proj.weight      torch.Size([9728, 2560])
# model.layers.9.mlp.up_proj.weight        torch.Size([9728, 2560])
# model.layers.9.mlp.down_proj.weight      torch.Size([2560, 9728])
# model.layers.9.input_layernorm.weight    torch.Size([2560])
# model.layers.9.post_attention_layernorm.weight   torch.Size([2560])
# model.layers.10.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.10.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.10.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.10.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.10.self_attn.q_norm.weight          torch.Size([128])
# model.layers.10.self_attn.k_norm.weight          torch.Size([128])
# model.layers.10.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.10.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.10.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.10.input_layernorm.weight   torch.Size([2560])
# model.layers.10.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.11.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.11.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.11.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.11.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.11.self_attn.q_norm.weight          torch.Size([128])
# model.layers.11.self_attn.k_norm.weight          torch.Size([128])
# model.layers.11.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.11.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.11.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.11.input_layernorm.weight   torch.Size([2560])
# model.layers.11.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.12.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.12.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.12.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.12.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.12.self_attn.q_norm.weight          torch.Size([128])
# model.layers.12.self_attn.k_norm.weight          torch.Size([128])
# model.layers.12.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.12.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.12.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.12.input_layernorm.weight   torch.Size([2560])
# model.layers.12.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.13.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.13.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.13.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.13.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.13.self_attn.q_norm.weight          torch.Size([128])
# model.layers.13.self_attn.k_norm.weight          torch.Size([128])
# model.layers.13.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.13.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.13.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.13.input_layernorm.weight   torch.Size([2560])
# model.layers.13.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.14.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.14.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.14.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.14.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.14.self_attn.q_norm.weight          torch.Size([128])
# model.layers.14.self_attn.k_norm.weight          torch.Size([128])
# model.layers.14.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.14.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.14.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.14.input_layernorm.weight   torch.Size([2560])
# model.layers.14.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.15.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.15.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.15.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.15.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.15.self_attn.q_norm.weight          torch.Size([128])
# model.layers.15.self_attn.k_norm.weight          torch.Size([128])
# model.layers.15.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.15.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.15.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.15.input_layernorm.weight   torch.Size([2560])
# model.layers.15.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.16.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.16.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.16.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.16.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.16.self_attn.q_norm.weight          torch.Size([128])
# model.layers.16.self_attn.k_norm.weight          torch.Size([128])
# model.layers.16.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.16.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.16.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.16.input_layernorm.weight   torch.Size([2560])
# model.layers.16.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.17.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.17.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.17.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.17.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.17.self_attn.q_norm.weight          torch.Size([128])
# model.layers.17.self_attn.k_norm.weight          torch.Size([128])
# model.layers.17.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.17.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.17.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.17.input_layernorm.weight   torch.Size([2560])
# model.layers.17.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.18.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.18.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.18.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.18.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.18.self_attn.q_norm.weight          torch.Size([128])
# model.layers.18.self_attn.k_norm.weight          torch.Size([128])
# model.layers.18.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.18.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.18.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.18.input_layernorm.weight   torch.Size([2560])
# model.layers.18.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.19.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.19.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.19.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.19.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.19.self_attn.q_norm.weight          torch.Size([128])
# model.layers.19.self_attn.k_norm.weight          torch.Size([128])
# model.layers.19.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.19.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.19.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.19.input_layernorm.weight   torch.Size([2560])
# model.layers.19.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.20.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.20.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.20.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.20.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.20.self_attn.q_norm.weight          torch.Size([128])
# model.layers.20.self_attn.k_norm.weight          torch.Size([128])
# model.layers.20.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.20.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.20.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.20.input_layernorm.weight   torch.Size([2560])
# model.layers.20.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.21.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.21.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.21.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.21.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.21.self_attn.q_norm.weight          torch.Size([128])
# model.layers.21.self_attn.k_norm.weight          torch.Size([128])
# model.layers.21.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.21.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.21.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.21.input_layernorm.weight   torch.Size([2560])
# model.layers.21.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.22.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.22.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.22.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.22.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.22.self_attn.q_norm.weight          torch.Size([128])
# model.layers.22.self_attn.k_norm.weight          torch.Size([128])
# model.layers.22.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.22.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.22.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.22.input_layernorm.weight   torch.Size([2560])
# model.layers.22.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.23.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.23.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.23.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.23.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.23.self_attn.q_norm.weight          torch.Size([128])
# model.layers.23.self_attn.k_norm.weight          torch.Size([128])
# model.layers.23.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.23.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.23.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.23.input_layernorm.weight   torch.Size([2560])
# model.layers.23.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.24.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.24.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.24.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.24.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.24.self_attn.q_norm.weight          torch.Size([128])
# model.layers.24.self_attn.k_norm.weight          torch.Size([128])
# model.layers.24.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.24.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.24.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.24.input_layernorm.weight   torch.Size([2560])
# model.layers.24.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.25.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.25.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.25.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.25.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.25.self_attn.q_norm.weight          torch.Size([128])
# model.layers.25.self_attn.k_norm.weight          torch.Size([128])
# model.layers.25.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.25.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.25.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.25.input_layernorm.weight   torch.Size([2560])
# model.layers.25.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.26.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.26.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.26.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.26.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.26.self_attn.q_norm.weight          torch.Size([128])
# model.layers.26.self_attn.k_norm.weight          torch.Size([128])
# model.layers.26.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.26.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.26.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.26.input_layernorm.weight   torch.Size([2560])
# model.layers.26.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.27.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.27.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.27.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.27.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.27.self_attn.q_norm.weight          torch.Size([128])
# model.layers.27.self_attn.k_norm.weight          torch.Size([128])
# model.layers.27.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.27.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.27.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.27.input_layernorm.weight   torch.Size([2560])
# model.layers.27.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.28.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.28.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.28.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.28.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.28.self_attn.q_norm.weight          torch.Size([128])
# model.layers.28.self_attn.k_norm.weight          torch.Size([128])
# model.layers.28.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.28.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.28.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.28.input_layernorm.weight   torch.Size([2560])
# model.layers.28.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.29.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.29.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.29.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.29.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.29.self_attn.q_norm.weight          torch.Size([128])
# model.layers.29.self_attn.k_norm.weight          torch.Size([128])
# model.layers.29.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.29.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.29.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.29.input_layernorm.weight   torch.Size([2560])
# model.layers.29.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.30.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.30.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.30.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.30.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.30.self_attn.q_norm.weight          torch.Size([128])
# model.layers.30.self_attn.k_norm.weight          torch.Size([128])
# model.layers.30.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.30.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.30.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.30.input_layernorm.weight   torch.Size([2560])
# model.layers.30.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.31.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.31.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.31.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.31.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.31.self_attn.q_norm.weight          torch.Size([128])
# model.layers.31.self_attn.k_norm.weight          torch.Size([128])
# model.layers.31.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.31.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.31.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.31.input_layernorm.weight   torch.Size([2560])
# model.layers.31.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.32.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.32.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.32.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.32.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.32.self_attn.q_norm.weight          torch.Size([128])
# model.layers.32.self_attn.k_norm.weight          torch.Size([128])
# model.layers.32.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.32.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.32.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.32.input_layernorm.weight   torch.Size([2560])
# model.layers.32.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.33.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.33.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.33.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.33.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.33.self_attn.q_norm.weight          torch.Size([128])
# model.layers.33.self_attn.k_norm.weight          torch.Size([128])
# model.layers.33.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.33.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.33.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.33.input_layernorm.weight   torch.Size([2560])
# model.layers.33.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.34.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.34.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.34.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.34.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.34.self_attn.q_norm.weight          torch.Size([128])
# model.layers.34.self_attn.k_norm.weight          torch.Size([128])
# model.layers.34.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.34.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.34.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.34.input_layernorm.weight   torch.Size([2560])
# model.layers.34.post_attention_layernorm.weight          torch.Size([2560])
# model.layers.35.self_attn.q_proj.weight          torch.Size([4096, 2560])
# model.layers.35.self_attn.k_proj.weight          torch.Size([1024, 2560])
# model.layers.35.self_attn.v_proj.weight          torch.Size([1024, 2560])
# model.layers.35.self_attn.o_proj.weight          torch.Size([2560, 4096])
# model.layers.35.self_attn.q_norm.weight          torch.Size([128])
# model.layers.35.self_attn.k_norm.weight          torch.Size([128])
# model.layers.35.mlp.gate_proj.weight     torch.Size([9728, 2560])
# model.layers.35.mlp.up_proj.weight       torch.Size([9728, 2560])
# model.layers.35.mlp.down_proj.weight     torch.Size([2560, 9728])
# model.layers.35.input_layernorm.weight   torch.Size([2560])
# model.layers.35.post_attention_layernorm.weight          torch.Size([2560])
# model.norm.weight        torch.Size([2560])
# lm_head.weight   torch.Size([151936, 2560])
# model.embed_tokens.weight: 136427342131264
# model.layers.0.self_attn.q_proj.weight: 136430466625600
# model.layers.0.self_attn.k_proj.weight: 671138496
# model.layers.0.self_attn.v_proj.weight: 779043968
# model.layers.0.self_attn.o_proj.weight: 805749760
# model.layers.0.self_attn.q_norm.weight: 665308544
# model.layers.0.self_attn.k_norm.weight: 669344640
# model.layers.0.mlp.gate_proj.weight: 136430175117376
# model.layers.0.mlp.up_proj.weight: 136430075498560
# model.layers.0.mlp.down_proj.weight: 136430274736192
# model.layers.0.input_layernorm.weight: 670396224
# model.layers.0.post_attention_layernorm.weight: 757353920
# model.layers.1.self_attn.q_proj.weight: 136429588951104
# model.layers.1.self_attn.k_proj.weight: 706480320
# model.layers.1.self_attn.v_proj.weight: 716966144
# model.layers.1.self_attn.o_proj.weight: 136429630898240
# model.layers.1.self_attn.q_norm.weight: 750724928
# model.layers.1.self_attn.k_norm.weight: 748877696
# model.layers.1.mlp.gate_proj.weight: 136429772464192
# model.layers.1.mlp.up_proj.weight: 136429672845376
# model.layers.1.mlp.down_proj.weight: 136429872083008
# model.layers.1.input_layernorm.weight: 757364224
# model.layers.1.post_attention_layernorm.weight: 670319168
# model.layers.2.self_attn.q_proj.weight: 136425103298624
# model.layers.2.self_attn.k_proj.weight: 963036928
# model.layers.2.self_attn.v_proj.weight: 973522752
# model.layers.2.self_attn.o_proj.weight: 136425145245760
# model.layers.2.self_attn.q_norm.weight: 746068864
# model.layers.2.self_attn.k_norm.weight: 745198464
# model.layers.2.mlp.gate_proj.weight: 136425286811712
# model.layers.2.mlp.up_proj.weight: 136425187192896
# model.layers.2.mlp.down_proj.weight: 136425386430528
# model.layers.2.input_layernorm.weight: 757319616
# model.layers.2.post_attention_layernorm.weight: 757444032
# model.layers.3.self_attn.q_proj.weight: 136420740968512
# model.layers.3.self_attn.k_proj.weight: 984008576
# model.layers.3.self_attn.v_proj.weight: 994494400
# model.layers.3.self_attn.o_proj.weight: 136424961732672
# model.layers.3.self_attn.q_norm.weight: 748686016
# model.layers.3.self_attn.k_norm.weight: 743243584
# model.layers.3.mlp.gate_proj.weight: 136420882534464
# model.layers.3.mlp.up_proj.weight: 136420782915648
# model.layers.3.mlp.down_proj.weight: 136425003679808
# model.layers.3.input_layernorm.weight: 757454336
# model.layers.3.post_attention_layernorm.weight: 757464640
# model.layers.4.self_attn.q_proj.weight: 136420358217792
# model.layers.4.self_attn.k_proj.weight: 1004980224
# model.layers.4.self_attn.v_proj.weight: 1015466048
# model.layers.4.self_attn.o_proj.weight: 136420400164928
# model.layers.4.self_attn.q_norm.weight: 743230400
# model.layers.4.self_attn.k_norm.weight: 746361536
# model.layers.4.mlp.gate_proj.weight: 136420541730880
# model.layers.4.mlp.up_proj.weight: 136420442112064
# model.layers.4.mlp.down_proj.weight: 136420641349696
# model.layers.4.input_layernorm.weight: 757474944
# model.layers.4.post_attention_layernorm.weight: 757485248
# model.layers.5.self_attn.q_proj.weight: 136419975467072
# model.layers.5.self_attn.k_proj.weight: 1025951872
# model.layers.5.self_attn.v_proj.weight: 1036437696
# model.layers.5.self_attn.o_proj.weight: 136420017414208
# model.layers.5.self_attn.q_norm.weight: 746577792
# model.layers.5.self_attn.k_norm.weight: 740243712
# model.layers.5.mlp.gate_proj.weight: 136420158980160
# model.layers.5.mlp.up_proj.weight: 136420059361344
# model.layers.5.mlp.down_proj.weight: 136420258598976
# model.layers.5.input_layernorm.weight: 768919552
# model.layers.5.post_attention_layernorm.weight: 768929856
# model.layers.6.self_attn.q_proj.weight: 136419592716352
# model.layers.6.self_attn.k_proj.weight: 1046923520
# model.layers.6.self_attn.v_proj.weight: 1057409344
# model.layers.6.self_attn.o_proj.weight: 136419634663488
# model.layers.6.self_attn.q_norm.weight: 746742400
# model.layers.6.self_attn.k_norm.weight: 743875968
# model.layers.6.mlp.gate_proj.weight: 136419776229440
# model.layers.6.mlp.up_proj.weight: 136419676610624
# model.layers.6.mlp.down_proj.weight: 136419875848256
# model.layers.6.input_layernorm.weight: 768940160
# model.layers.6.post_attention_layernorm.weight: 768950464
# model.layers.7.self_attn.q_proj.weight: 136419209965632
# model.layers.7.self_attn.k_proj.weight: 1067895168
# model.layers.7.self_attn.v_proj.weight: 1078380992
# model.layers.7.self_attn.o_proj.weight: 136419251912768
# model.layers.7.self_attn.q_norm.weight: 741521408
# model.layers.7.self_attn.k_norm.weight: 748893376
# model.layers.7.mlp.gate_proj.weight: 136419393478720
# model.layers.7.mlp.up_proj.weight: 136419293859904
# model.layers.7.mlp.down_proj.weight: 136419493097536
# model.layers.7.input_layernorm.weight: 768960768
# model.layers.7.post_attention_layernorm.weight: 651132480
# model.layers.8.self_attn.q_proj.weight: 136418827214912
# model.layers.8.self_attn.k_proj.weight: 1088866816
# model.layers.8.self_attn.v_proj.weight: 1099352640
# model.layers.8.self_attn.o_proj.weight: 136418869162048
# model.layers.8.self_attn.q_norm.weight: 739524288
# model.layers.8.self_attn.k_norm.weight: 747738048
# model.layers.8.mlp.gate_proj.weight: 136419010728000
# model.layers.8.mlp.up_proj.weight: 136418911109184
# model.layers.8.mlp.down_proj.weight: 136419110346816
# model.layers.8.input_layernorm.weight: 651142784
# model.layers.8.post_attention_layernorm.weight: 651153088
# model.layers.9.self_attn.q_proj.weight: 136418444464192
# model.layers.9.self_attn.k_proj.weight: 1109838464
# model.layers.9.self_attn.v_proj.weight: 1120324288
# model.layers.9.self_attn.o_proj.weight: 136418486411328
# model.layers.9.self_attn.q_norm.weight: 743298944
# model.layers.9.self_attn.k_norm.weight: 747037184
# model.layers.9.mlp.gate_proj.weight: 136418627977280
# model.layers.9.mlp.up_proj.weight: 136418528358464
# model.layers.9.mlp.down_proj.weight: 136418727596096
# model.layers.9.input_layernorm.weight: 651163392
# model.layers.9.post_attention_layernorm.weight: 651173696
# model.layers.10.self_attn.q_proj.weight: 136427300184128
# model.layers.10.self_attn.k_proj.weight: 727451968
# model.layers.10.self_attn.v_proj.weight: 847692864
# model.layers.10.self_attn.o_proj.weight: 136429447385152
# model.layers.10.self_attn.q_norm.weight: 744112768
# model.layers.10.self_attn.k_norm.weight: 740908352
# model.layers.10.mlp.gate_proj.weight: 136429200994368
# model.layers.10.mlp.up_proj.weight: 136428999667776
# model.layers.10.mlp.down_proj.weight: 136429489332288
# model.layers.10.input_layernorm.weight: 670329472
# model.layers.10.post_attention_layernorm.weight: 670339776
# model.layers.11.self_attn.q_proj.weight: 136426917433408
# model.layers.11.self_attn.k_proj.weight: 858178688
# model.layers.11.self_attn.v_proj.weight: 868664512
# model.layers.11.self_attn.o_proj.weight: 136426959380544
# model.layers.11.self_attn.q_norm.weight: 742650880
# model.layers.11.self_attn.k_norm.weight: 747608896
# model.layers.11.mlp.gate_proj.weight: 136427100946496
# model.layers.11.mlp.up_proj.weight: 136427001327680
# model.layers.11.mlp.down_proj.weight: 136427200565312
# model.layers.11.input_layernorm.weight: 651191552
# model.layers.11.post_attention_layernorm.weight: 651201856
# model.layers.12.self_attn.q_proj.weight: 136426534682688
# model.layers.12.self_attn.k_proj.weight: 879150336
# model.layers.12.self_attn.v_proj.weight: 889636160
# model.layers.12.self_attn.o_proj.weight: 136426576629824
# model.layers.12.self_attn.q_norm.weight: 749250048
# model.layers.12.self_attn.k_norm.weight: 744615552
# model.layers.12.mlp.gate_proj.weight: 136426718195776
# model.layers.12.mlp.up_proj.weight: 136426618576960
# model.layers.12.mlp.down_proj.weight: 136426817814592
# model.layers.12.input_layernorm.weight: 651212160
# model.layers.12.post_attention_layernorm.weight: 651420160
# model.layers.13.self_attn.q_proj.weight: 136426151931968
# model.layers.13.self_attn.k_proj.weight: 900121984
# model.layers.13.self_attn.v_proj.weight: 910607808
# model.layers.13.self_attn.o_proj.weight: 136426193879104
# model.layers.13.self_attn.q_norm.weight: 747829888
# model.layers.13.self_attn.k_norm.weight: 741827968
# model.layers.13.mlp.gate_proj.weight: 136426335445056
# model.layers.13.mlp.up_proj.weight: 136426235826240
# model.layers.13.mlp.down_proj.weight: 136426435063872
# model.layers.13.input_layernorm.weight: 651430464
# model.layers.13.post_attention_layernorm.weight: 651440768
# model.layers.14.self_attn.q_proj.weight: 136425769181248
# model.layers.14.self_attn.k_proj.weight: 921093632
# model.layers.14.self_attn.v_proj.weight: 931579456
# model.layers.14.self_attn.o_proj.weight: 136425811128384
# model.layers.14.self_attn.q_norm.weight: 747562880
# model.layers.14.self_attn.k_norm.weight: 739885376
# model.layers.14.mlp.gate_proj.weight: 136425952694336
# model.layers.14.mlp.up_proj.weight: 136425853075520
# model.layers.14.mlp.down_proj.weight: 136426052313152
# model.layers.14.input_layernorm.weight: 757299008
# model.layers.14.post_attention_layernorm.weight: 757309312
# model.layers.15.self_attn.q_proj.weight: 136425486049344
# model.layers.15.self_attn.k_proj.weight: 942065280
# model.layers.15.self_attn.v_proj.weight: 952551104
# model.layers.15.self_attn.o_proj.weight: 136425527996480
# model.layers.15.self_attn.q_norm.weight: 749379904
# model.layers.15.self_attn.k_norm.weight: 742912768
# model.layers.15.mlp.gate_proj.weight: 136425669562432
# model.layers.15.mlp.up_proj.weight: 136425569943616
# model.layers.15.mlp.down_proj.weight: 136424862113856
# model.layers.15.input_layernorm.weight: 670383680
# model.layers.15.post_attention_layernorm.weight: 651254592
# model.layers.16.self_attn.q_proj.weight: 136424479363136
# model.layers.16.self_attn.k_proj.weight: 1130810112
# model.layers.16.self_attn.v_proj.weight: 1141295936
# model.layers.16.self_attn.o_proj.weight: 136424521310272
# model.layers.16.self_attn.q_norm.weight: 741639104
# model.layers.16.self_attn.k_norm.weight: 749295744
# model.layers.16.mlp.gate_proj.weight: 136424662876224
# model.layers.16.mlp.up_proj.weight: 136424563257408
# model.layers.16.mlp.down_proj.weight: 136424762495040
# model.layers.16.input_layernorm.weight: 651264896
# model.layers.16.post_attention_layernorm.weight: 651275200
# model.layers.17.self_attn.q_proj.weight: 136424096612416
# model.layers.17.self_attn.k_proj.weight: 1151781760
# model.layers.17.self_attn.v_proj.weight: 1162267584
# model.layers.17.self_attn.o_proj.weight: 136424138559552
# model.layers.17.self_attn.q_norm.weight: 748008576
# model.layers.17.self_attn.k_norm.weight: 739204288
# model.layers.17.mlp.gate_proj.weight: 136424280125504
# model.layers.17.mlp.up_proj.weight: 136424180506688
# model.layers.17.mlp.down_proj.weight: 136424379744320
# model.layers.17.input_layernorm.weight: 651285504
# model.layers.17.post_attention_layernorm.weight: 651295808
# model.layers.18.self_attn.q_proj.weight: 136423713861696
# model.layers.18.self_attn.k_proj.weight: 1172753408
# model.layers.18.self_attn.v_proj.weight: 1183239232
# model.layers.18.self_attn.o_proj.weight: 136423755808832
# model.layers.18.self_attn.q_norm.weight: 739863744
# model.layers.18.self_attn.k_norm.weight: 744473728
# model.layers.18.mlp.gate_proj.weight: 136423897374784
# model.layers.18.mlp.up_proj.weight: 136423797755968
# model.layers.18.mlp.down_proj.weight: 136423996993600
# model.layers.18.input_layernorm.weight: 767607168
# model.layers.18.post_attention_layernorm.weight: 767617472
# model.layers.19.self_attn.q_proj.weight: 136423331110976
# model.layers.19.self_attn.k_proj.weight: 1193725056
# model.layers.19.self_attn.v_proj.weight: 1204210880
# model.layers.19.self_attn.o_proj.weight: 136423373058112
# model.layers.19.self_attn.q_norm.weight: 745125120
# model.layers.19.self_attn.k_norm.weight: 742343872
# model.layers.19.mlp.gate_proj.weight: 136423514624064
# model.layers.19.mlp.up_proj.weight: 136423415005248
# model.layers.19.mlp.down_proj.weight: 136423614242880
# model.layers.19.input_layernorm.weight: 767627776
# model.layers.19.post_attention_layernorm.weight: 767638080
# model.layers.20.self_attn.q_proj.weight: 136422948360256
# model.layers.20.self_attn.k_proj.weight: 1214696704
# model.layers.20.self_attn.v_proj.weight: 1225182528
# model.layers.20.self_attn.o_proj.weight: 136422990307392
# model.layers.20.self_attn.q_norm.weight: 748999808
# model.layers.20.self_attn.k_norm.weight: 740822336
# model.layers.20.mlp.gate_proj.weight: 136423131873344
# model.layers.20.mlp.up_proj.weight: 136423032254528
# model.layers.20.mlp.down_proj.weight: 136423231492160
# model.layers.20.input_layernorm.weight: 767648384
# model.layers.20.post_attention_layernorm.weight: 767658688
# model.layers.21.self_attn.q_proj.weight: 136422565609536
# model.layers.21.self_attn.k_proj.weight: 1235668352
# model.layers.21.self_attn.v_proj.weight: 1246154176
# model.layers.21.self_attn.o_proj.weight: 136422607556672
# model.layers.21.self_attn.q_norm.weight: 748195840
# model.layers.21.self_attn.k_norm.weight: 744203776
# model.layers.21.mlp.gate_proj.weight: 136422749122624
# model.layers.21.mlp.up_proj.weight: 136422649503808
# model.layers.21.mlp.down_proj.weight: 136422848741440
# model.layers.21.input_layernorm.weight: 767668992
# model.layers.21.post_attention_layernorm.weight: 767679296
# model.layers.22.self_attn.q_proj.weight: 136422182858816
# model.layers.22.self_attn.k_proj.weight: 1256640000
# model.layers.22.self_attn.v_proj.weight: 1267125824
# model.layers.22.self_attn.o_proj.weight: 136422224805952
# model.layers.22.self_attn.q_norm.weight: 743954944
# model.layers.22.self_attn.k_norm.weight: 739717056
# model.layers.22.mlp.gate_proj.weight: 136422366371904
# model.layers.22.mlp.up_proj.weight: 136422266753088
# model.layers.22.mlp.down_proj.weight: 136422465990720
# model.layers.22.input_layernorm.weight: 767689600
# model.layers.22.post_attention_layernorm.weight: 767699904
# model.layers.23.self_attn.q_proj.weight: 136421800108096
# model.layers.23.self_attn.k_proj.weight: 1277611648
# model.layers.23.self_attn.v_proj.weight: 1288097472
# model.layers.23.self_attn.o_proj.weight: 136421842055232
# model.layers.23.self_attn.q_norm.weight: 740708672
# model.layers.23.self_attn.k_norm.weight: 747993088
# model.layers.23.mlp.gate_proj.weight: 136421983621184
# model.layers.23.mlp.up_proj.weight: 136421884002368
# model.layers.23.mlp.down_proj.weight: 136422083240000
# model.layers.23.input_layernorm.weight: 767710208
# model.layers.23.post_attention_layernorm.weight: 767325376
# model.layers.24.self_attn.q_proj.weight: 136421417357376
# model.layers.24.self_attn.k_proj.weight: 1298583296
# model.layers.24.self_attn.v_proj.weight: 1309069120
# model.layers.24.self_attn.o_proj.weight: 136421459304512
# model.layers.24.self_attn.q_norm.weight: 744088960
# model.layers.24.self_attn.k_norm.weight: 744852288
# model.layers.24.mlp.gate_proj.weight: 136421600870464
# model.layers.24.mlp.up_proj.weight: 136421501251648
# model.layers.24.mlp.down_proj.weight: 136421700489280
# model.layers.24.input_layernorm.weight: 767335680
# model.layers.24.post_attention_layernorm.weight: 767345984
# model.layers.25.self_attn.q_proj.weight: 136421034606656
# model.layers.25.self_attn.k_proj.weight: 1319554944
# model.layers.25.self_attn.v_proj.weight: 1330040768
# model.layers.25.self_attn.o_proj.weight: 136421076553792
# model.layers.25.self_attn.q_norm.weight: 748953792
# model.layers.25.self_attn.k_norm.weight: 739423872
# model.layers.25.mlp.gate_proj.weight: 136421218119744
# model.layers.25.mlp.up_proj.weight: 136421118500928
# model.layers.25.mlp.down_proj.weight: 136421317738560
# model.layers.25.input_layernorm.weight: 767356288
# model.layers.25.post_attention_layernorm.weight: 767366592
# model.layers.26.self_attn.q_proj.weight: 136418103660608
# model.layers.26.self_attn.k_proj.weight: 1340526592
# model.layers.26.self_attn.v_proj.weight: 1351012416
# model.layers.26.self_attn.o_proj.weight: 136420992659520
# model.layers.26.self_attn.q_norm.weight: 743210880
# model.layers.26.self_attn.k_norm.weight: 741629952
# model.layers.26.mlp.gate_proj.weight: 136418245226560
# model.layers.26.mlp.up_proj.weight: 136418145607744
# model.layers.26.mlp.down_proj.weight: 136418344845376
# model.layers.26.input_layernorm.weight: 767376896
# model.layers.26.post_attention_layernorm.weight: 767387200
# model.layers.27.self_attn.q_proj.weight: 136417720909888
# model.layers.27.self_attn.k_proj.weight: 1361498240
# model.layers.27.self_attn.v_proj.weight: 1371984064
# model.layers.27.self_attn.o_proj.weight: 136417762857024
# model.layers.27.self_attn.q_norm.weight: 748149760
# model.layers.27.self_attn.k_norm.weight: 747672512
# model.layers.27.mlp.gate_proj.weight: 136417904422976
# model.layers.27.mlp.up_proj.weight: 136417804804160
# model.layers.27.mlp.down_proj.weight: 136418004041792
# model.layers.27.input_layernorm.weight: 767397504
# model.layers.27.post_attention_layernorm.weight: 767407808
# model.layers.28.self_attn.q_proj.weight: 136417338159168
# model.layers.28.self_attn.k_proj.weight: 1382469888
# model.layers.28.self_attn.v_proj.weight: 1392955712
# model.layers.28.self_attn.o_proj.weight: 136417380106304
# model.layers.28.self_attn.q_norm.weight: 739970112
# model.layers.28.self_attn.k_norm.weight: 750654016
# model.layers.28.mlp.gate_proj.weight: 136417521672256
# model.layers.28.mlp.up_proj.weight: 136417422053440
# model.layers.28.mlp.down_proj.weight: 136417621291072
# model.layers.28.input_layernorm.weight: 767418112
# model.layers.28.post_attention_layernorm.weight: 767428416
# model.layers.29.self_attn.q_proj.weight: 136416955408448
# model.layers.29.self_attn.k_proj.weight: 1403441536
# model.layers.29.self_attn.v_proj.weight: 1413927360
# model.layers.29.self_attn.o_proj.weight: 136416997355584
# model.layers.29.self_attn.q_norm.weight: 750013632
# model.layers.29.self_attn.k_norm.weight: 744638336
# model.layers.29.mlp.gate_proj.weight: 136417138921536
# model.layers.29.mlp.up_proj.weight: 136417039302720
# model.layers.29.mlp.down_proj.weight: 136417238540352
# model.layers.29.input_layernorm.weight: 767438720
# model.layers.29.post_attention_layernorm.weight: 767449024
# model.layers.30.self_attn.q_proj.weight: 136416572657728
# model.layers.30.self_attn.k_proj.weight: 1424413184
# model.layers.30.self_attn.v_proj.weight: 1434899008
# model.layers.30.self_attn.o_proj.weight: 136416614604864
# model.layers.30.self_attn.q_norm.weight: 748000832
# model.layers.30.self_attn.k_norm.weight: 741030016
# model.layers.30.mlp.gate_proj.weight: 136416756170816
# model.layers.30.mlp.up_proj.weight: 136416656552000
# model.layers.30.mlp.down_proj.weight: 136416855789632
# model.layers.30.input_layernorm.weight: 651464192
# model.layers.30.post_attention_layernorm.weight: 651474496
# model.layers.31.self_attn.q_proj.weight: 136416189907008
# model.layers.31.self_attn.k_proj.weight: 1445384832
# model.layers.31.self_attn.v_proj.weight: 1455870656
# model.layers.31.self_attn.o_proj.weight: 136416231854144
# model.layers.31.self_attn.q_norm.weight: 749165696
# model.layers.31.self_attn.k_norm.weight: 739295680
# model.layers.31.mlp.gate_proj.weight: 136416373420096
# model.layers.31.mlp.up_proj.weight: 136416273801280
# model.layers.31.mlp.down_proj.weight: 136416473038912
# model.layers.31.input_layernorm.weight: 651484800
# model.layers.31.post_attention_layernorm.weight: 651495104
# model.layers.32.self_attn.q_proj.weight: 136415807156288
# model.layers.32.self_attn.k_proj.weight: 1466356480
# model.layers.32.self_attn.v_proj.weight: 1476842304
# model.layers.32.self_attn.o_proj.weight: 136415849103424
# model.layers.32.self_attn.q_norm.weight: 742940032
# model.layers.32.self_attn.k_norm.weight: 745981952
# model.layers.32.mlp.gate_proj.weight: 136415990669376
# model.layers.32.mlp.up_proj.weight: 136415891050560
# model.layers.32.mlp.down_proj.weight: 136416090288192
# model.layers.32.input_layernorm.weight: 651505408
# model.layers.32.post_attention_layernorm.weight: 651515712
# model.layers.33.self_attn.q_proj.weight: 136415424405568
# model.layers.33.self_attn.k_proj.weight: 1487328128
# model.layers.33.self_attn.v_proj.weight: 1497813952
# model.layers.33.self_attn.o_proj.weight: 136415466352704
# model.layers.33.self_attn.q_norm.weight: 740195648
# model.layers.33.self_attn.k_norm.weight: 749294976
# model.layers.33.mlp.gate_proj.weight: 136415607918656
# model.layers.33.mlp.up_proj.weight: 136415508299840
# model.layers.33.mlp.down_proj.weight: 136415707537472
# model.layers.33.input_layernorm.weight: 651526016
# model.layers.33.post_attention_layernorm.weight: 651536320
# model.layers.34.self_attn.q_proj.weight: 136415041654848
# model.layers.34.self_attn.k_proj.weight: 1508299776
# model.layers.34.self_attn.v_proj.weight: 1518785600
# model.layers.34.self_attn.o_proj.weight: 136415083601984
# model.layers.34.self_attn.q_norm.weight: 741950400
# model.layers.34.self_attn.k_norm.weight: 744251904
# model.layers.34.mlp.gate_proj.weight: 136415225167936
# model.layers.34.mlp.up_proj.weight: 136415125549120
# model.layers.34.mlp.down_proj.weight: 136415324786752
# model.layers.34.input_layernorm.weight: 651546624
# model.layers.34.post_attention_layernorm.weight: 651556928
# model.layers.35.self_attn.q_proj.weight: 136414858141760
# model.layers.35.self_attn.k_proj.weight: 1529271424
# model.layers.35.self_attn.v_proj.weight: 1539757248
# model.layers.35.self_attn.o_proj.weight: 136414900088896
# model.layers.35.self_attn.q_norm.weight: 743963328
# model.layers.35.self_attn.k_norm.weight: 747346112
# model.layers.35.mlp.gate_proj.weight: 136414942036032
# model.layers.35.mlp.up_proj.weight: 136414559260736
# model.layers.35.mlp.down_proj.weight: 136414758522944
# model.layers.35.input_layernorm.weight: 651238784
# model.layers.35.post_attention_layernorm.weight: 651567232
# model.norm.weight: 651577536
# lm_head.weight: 136427342131264
