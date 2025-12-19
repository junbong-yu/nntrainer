import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

def save_flan_t5_weights(params, config, dtype, file):
    """Convert and save weights as numpy array format for T5 model"""

    def save_weight(weight_name, is_transpose=False):
        """Save weight with optional transpose"""
        if is_transpose:
            print(f"Saving transposed weight: {weight_name}, shape: {params[weight_name].permute(1, 0).shape}")
            np.array(params[weight_name].permute(1, 0).float(), dtype=dtype).tofile(file)
        else:
            print(f"Saving weight: {weight_name}, shape: {params[weight_name].shape}")
            np.array(params[weight_name].float(), dtype=dtype).tofile(file)

    # Save shared embedding weight
    save_weight("shared.weight")

    # Save encoder weights
    n_encoder_layers = config.num_layers
    for layer_idx in range(n_encoder_layers):
        layer_prefix = f"encoder.block.{layer_idx}."

        # Self attention layer
        save_weight(f"{layer_prefix}layer.0.SelfAttention.q.weight", True)
        save_weight(f"{layer_prefix}layer.0.SelfAttention.k.weight", True)
        save_weight(f"{layer_prefix}layer.0.SelfAttention.v.weight", True)
        save_weight(f"{layer_prefix}layer.0.SelfAttention.o.weight", True)

        # Relative attention bias (no transpose)
        if f"{layer_prefix}layer.0.SelfAttention.relative_attention_bias.weight" in params:
            save_weight(f"{layer_prefix}layer.0.SelfAttention.relative_attention_bias.weight")

        # Layer norm (no transpose)
        save_weight(f"{layer_prefix}layer.0.layer_norm.weight")

        # Feed forward layer
        save_weight(f"{layer_prefix}layer.1.DenseReluDense.wi_0.weight", True)
        save_weight(f"{layer_prefix}layer.1.DenseReluDense.wi_1.weight", True)
        save_weight(f"{layer_prefix}layer.1.DenseReluDense.wo.weight", True)

        # Layer norm (no transpose)
        save_weight(f"{layer_prefix}layer.1.layer_norm.weight")

    # Encoder final layer norm
    save_weight("encoder.final_layer_norm.weight")

    # Save decoder weights
    n_decoder_layers = config.num_decoder_layers
    for layer_idx in range(n_decoder_layers):
        layer_prefix = f"decoder.block.{layer_idx}."

        # Self attention layer
        save_weight(f"{layer_prefix}layer.0.SelfAttention.q.weight", True)
        save_weight(f"{layer_prefix}layer.0.SelfAttention.k.weight", True)
        save_weight(f"{layer_prefix}layer.0.SelfAttention.v.weight", True)
        save_weight(f"{layer_prefix}layer.0.SelfAttention.o.weight", True)

        # Relative attention bias (no transpose)
        if f"{layer_prefix}layer.0.SelfAttention.relative_attention_bias.weight" in params:
            save_weight(f"{layer_prefix}layer.0.SelfAttention.relative_attention_bias.weight")

        # Layer norm (no transpose)
        save_weight(f"{layer_prefix}layer.0.layer_norm.weight")

        # Encoder-decoder attention layer
        save_weight(f"{layer_prefix}layer.1.EncDecAttention.q.weight", True)
        save_weight(f"{layer_prefix}layer.1.EncDecAttention.k.weight", True)
        save_weight(f"{layer_prefix}layer.1.EncDecAttention.v.weight", True)
        save_weight(f"{layer_prefix}layer.1.EncDecAttention.o.weight", True)

        # Layer norm (no transpose)
        save_weight(f"{layer_prefix}layer.1.layer_norm.weight")

        # Feed forward layer
        save_weight(f"{layer_prefix}layer.2.DenseReluDense.wi_0.weight", True)
        save_weight(f"{layer_prefix}layer.2.DenseReluDense.wi_1.weight", True)
        save_weight(f"{layer_prefix}layer.2.DenseReluDense.wo.weight", True)

        # Layer norm (no transpose)
        save_weight(f"{layer_prefix}layer.2.layer_norm.weight")

    # Decoder final layer norm
    save_weight("decoder.final_layer_norm.weight")

    # Language model head
    save_weight("lm_head.weight", True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="google/flan-t5-small")
    parser.add_argument("--output_name", type=str, default="./flan_t5_small_weights.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    args = parser.parse_args()

    data_dtype = args.data_type
    model_path = args.model_path
    output_name = args.output_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    print(model)

    #print(model.state_dict())
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    for name, param in model.state_dict().items():
        print(f"{name}: {param.data_ptr()}")

    with open(output_name, "wb") as f_model:
        save_flan_t5_weights(model.state_dict(), config, data_dtype, f_model)
