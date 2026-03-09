// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   quantize.cpp
 * @date   09 March 2026
 * @brief  Quantization utility for XLMRoberta models.
 *         Reads a FP32 model and converts weights to a target data type,
 *         saving both the quantized .bin file and a new nntr_config.json.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @usage
 *   nntr_quantize_xlmroberta <model_path> [options]
 *
 *   Required:
 *     <model_path>        Path to the model directory containing:
 *                           config.json, generation_config.json,
 *                           nntr_config.json, and the .bin weight file.
 *
 *   Options:
 *     --output, -o <path> Output directory (default: <model_path>)
 *     --fc_dtype <type>   Target dtype for FC layers (default: Q4_0)
 *     --embd_dtype <type> Target dtype for embedding layer (default: FP32)
 *     --output_bin <name> Output bin filename (auto-generated if omitted)
 *
 *   Supported data types: FP32, FP16, Q4_0, Q6_K
 *
 *   Example:
 *     # Quantize XLMRoberta to Q4_0 FC layers (embedding stays FP32):
 *     nntr_quantize_xlmroberta /path/to/xlmroberta --fc_dtype Q4_0
 *
 *     # Quantize with Q6_K embedding and Q4_0 FC layers:
 *     nntr_quantize_xlmroberta /path/to/xlmroberta --fc_dtype Q4_0 --embd_dtype
 * Q6_K
 *
 *     # Quantize to a different output directory:
 *     nntr_quantize_xlmroberta /path/to/xlmroberta -o /output/xlmroberta-q4
 */

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>
#include <tensor_dim.h>

#include "causal_lm.h"
#include "xlm_roberta.h"

using json = nlohmann::json;
using DataType = ml::train::TensorDim::DataType;

namespace {

const std::map<std::string, DataType> dtype_str_map = {
  {"FP32", DataType::FP32}, {"FP16", DataType::FP16}, {"Q4_0", DataType::Q4_0},
  {"Q6_K", DataType::Q6_K}, {"Q4_K", DataType::Q4_K}, {"NONE", DataType::NONE},
};

DataType strToDataType(const std::string &s) {
  std::string upper = s;
  std::transform(upper.begin(), upper.end(), upper.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  auto it = dtype_str_map.find(upper);
  if (it == dtype_str_map.end()) {
    throw std::invalid_argument("Unsupported data type: " + s +
                                ". Supported: FP32, FP16, Q4_0, Q6_K, Q4_K");
  }
  return it->second;
}

std::string dataTypeToStr(DataType dt) {
  for (const auto &[key, val] : dtype_str_map) {
    if (val == dt)
      return key;
  }
  return "NONE";
}

std::string buildModelTensorType(const std::string &fc_dtype) {
  return fc_dtype + "-FP32";
}

std::string generateOutputBinName(const std::string &original_bin,
                                  const std::string &fc_dtype,
                                  const std::string &embd_dtype) {
  std::string base = original_bin;
  auto dot_pos = base.rfind(".bin");
  if (dot_pos != std::string::npos)
    base = base.substr(0, dot_pos);

  std::vector<std::string> dtype_suffixes = {"_fp32", "_fp16", "_q40", "_q4_0",
                                             "_q6k",  "_q6_k", "_q4k", "_q4_k"};
  for (const auto &suffix : dtype_suffixes) {
    auto pos = base.rfind(suffix);
    if (pos != std::string::npos && pos + suffix.size() == base.size()) {
      base = base.substr(0, pos);
      break;
    }
  }

  std::string fc_lower = fc_dtype;
  std::transform(fc_lower.begin(), fc_lower.end(), fc_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::string fc_clean = fc_lower;
  fc_clean.erase(std::remove(fc_clean.begin(), fc_clean.end(), '_'),
                 fc_clean.end());

  std::string embd_lower = embd_dtype;
  std::transform(embd_lower.begin(), embd_lower.end(), embd_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::string embd_clean = embd_lower;
  embd_clean.erase(std::remove(embd_clean.begin(), embd_clean.end(), '_'),
                   embd_clean.end());

  if (embd_clean == fc_clean) {
    return base + "_" + fc_clean + ".bin";
  }
  return base + "_" + fc_clean + "_embd" + embd_clean + ".bin";
}

void registerAllModels() {
  auto &factory = causallm::Factory::Instance();

  factory.registerModel("XLMRobertaModel",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<xlmroberta::XLMRoberta>(
                            cfg, generation_cfg, nntr_cfg);
                        });
}

void printUsage(const char *prog) {
  std::cout
    << "Usage: " << prog << " <model_path> [options]\n"
    << "\n"
    << "Quantize an XLMRoberta model from FP32 to a target data type.\n"
    << "\n"
    << "Required:\n"
    << "  <model_path>          Path to model directory containing:\n"
    << "                          config.json, generation_config.json,\n"
    << "                          nntr_config.json, and .bin weight file\n"
    << "\n"
    << "Options:\n"
    << "  --output, -o <path>   Output directory (default: <model_path>)\n"
    << "  --fc_dtype <type>     Target dtype for FC layers (default: Q4_0)\n"
    << "  --embd_dtype <type>   Target dtype for embedding (default: FP32)\n"
    << "  --output_bin <name>   Output .bin filename (auto-generated if "
       "omitted)\n"
    << "  --config <path>       Use a target nntr_config.json instead of\n"
    << "                        individual dtype options.\n"
    << "  --help, -h            Show this help message\n"
    << "\n"
    << "Supported data types: FP32, FP16, Q4_0, Q6_K, Q4_K\n"
    << "\n"
    << "Examples:\n"
    << "  # Quantize FC layers to Q4_0 (default):\n"
    << "  " << prog << " /path/to/xlmroberta\n"
    << "\n"
    << "  # Quantize FC layers to Q4_0 and embedding to Q6_K:\n"
    << "  " << prog
    << " /path/to/xlmroberta --fc_dtype Q4_0 --embd_dtype Q6_K\n"
    << "\n"
    << "  # Quantize to a different output directory:\n"
    << "  " << prog << " /path/to/xlmroberta -o /output/xlmroberta-q4\n"
    << "\n"
    << "  # Use a target nntr_config.json:\n"
    << "  " << prog
    << " /path/to/xlmroberta --config /path/to/target_nntr_config.json\n";
}

std::map<std::string, DataType>
buildLayerDtypeMap(int num_layers, DataType fc_dtype, DataType embd_dtype) {

  std::map<std::string, DataType> dtype_map;

  if (embd_dtype != DataType::FP32 && embd_dtype != DataType::NONE) {
    dtype_map["embedding_word_embedding"] = embd_dtype;
    dtype_map["embedding_position_embedding"] = embd_dtype;
    dtype_map["embedding_token_type_embedding"] = embd_dtype;
  }

  for (int i = 0; i < num_layers; ++i) {
    std::string prefix = "encoder_" + std::to_string(i);

    if (fc_dtype != DataType::FP32 && fc_dtype != DataType::NONE) {
      dtype_map[prefix + "_attention_self_wq"] = fc_dtype;
      dtype_map[prefix + "_attention_self_wk"] = fc_dtype;
      dtype_map[prefix + "_attention_self_wv"] = fc_dtype;
      dtype_map[prefix + "_attention_self_wo"] = fc_dtype;
      dtype_map[prefix + "_intermediate_dense"] = fc_dtype;
      dtype_map[prefix + "_output_dense"] = fc_dtype;
    }
  }

  if (fc_dtype != DataType::FP32 && fc_dtype != DataType::NONE) {
    dtype_map["pooler_dense"] = fc_dtype;
  }

  return dtype_map;
}

} // anonymous namespace

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }

  std::string first_arg = argv[1];
  if (first_arg == "--help" || first_arg == "-h") {
    printUsage(argv[0]);
    return EXIT_SUCCESS;
  }

  std::string model_path = argv[1];
  std::string output_dir = "";
  std::string fc_dtype_str = "Q4_0";
  std::string embd_dtype_str = "FP32";
  std::string output_bin_name = "";
  std::string target_config_path = "";

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "--fc_dtype" && i + 1 < argc) {
      fc_dtype_str = argv[++i];
    } else if (arg == "--embd_dtype" && i + 1 < argc) {
      embd_dtype_str = argv[++i];
    } else if (arg == "--output_bin" && i + 1 < argc) {
      output_bin_name = argv[++i];
    } else if (arg == "--config" && i + 1 < argc) {
      target_config_path = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      printUsage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  try {
    std::cout << "==========================================================\n";
    std::cout << "  NNTrainer XLMRoberta Quantization Utility\n";
    std::cout << "==========================================================\n";
    std::cout << "[1/5] Loading configurations from: " << model_path << "\n";

    json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    json generation_cfg =
      causallm::LoadJsonFile(model_path + "/generation_config.json");
    json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

    if (!target_config_path.empty()) {
      std::cout << "  Using target config: " << target_config_path << "\n";
      json target_cfg = causallm::LoadJsonFile(target_config_path);
      if (target_cfg.contains("fc_layer_dtype"))
        fc_dtype_str = target_cfg["fc_layer_dtype"].get<std::string>();
      if (target_cfg.contains("embedding_dtype"))
        embd_dtype_str = target_cfg["embedding_dtype"].get<std::string>();
      if (target_cfg.contains("model_file_name") && output_bin_name.empty())
        output_bin_name = target_cfg["model_file_name"].get<std::string>();
    }

    DataType fc_dtype = strToDataType(fc_dtype_str);
    DataType embd_dtype = strToDataType(embd_dtype_str);

    std::string src_tensor_type =
      nntr_cfg["model_tensor_type"].get<std::string>();
    if (src_tensor_type != "FP32-FP32") {
      std::cerr << "[WARNING] Source model_tensor_type is '" << src_tensor_type
                << "', not 'FP32-FP32'.\n"
                << "  Quantization from non-FP32 models may produce unexpected "
                   "results.\n";
    }

    if (output_dir.empty())
      output_dir = model_path;
    std::filesystem::create_directories(output_dir);

    std::string original_bin = nntr_cfg["model_file_name"].get<std::string>();
    if (output_bin_name.empty()) {
      output_bin_name = generateOutputBinName(
        original_bin, dataTypeToStr(fc_dtype), dataTypeToStr(embd_dtype));
    }

    std::string src_weight_path = model_path + "/" + original_bin;
    std::string dst_weight_path = output_dir + "/" + output_bin_name;

    int num_layers = cfg["num_hidden_layers"].get<int>();

    std::cout << "  Architecture: "
              << cfg["architectures"].get<std::vector<std::string>>()[0]
              << "\n";
    std::cout << "  Num layers:   " << num_layers << "\n";
    std::cout << "  Source:       " << src_weight_path << "\n";
    std::cout << "  Target:       " << dst_weight_path << "\n";
    std::cout << "  FC dtype:     " << dataTypeToStr(fc_dtype) << "\n";
    std::cout << "  Embed dtype:  " << dataTypeToStr(embd_dtype) << "\n";
    std::cout << "\n";

    std::cout << "[2/5] Creating and initializing model...\n";

    registerAllModels();

    std::string architecture =
      cfg["architectures"].get<std::vector<std::string>>()[0];

    auto model = causallm::Factory::Instance().create(architecture, cfg,
                                                      generation_cfg, nntr_cfg);
    if (!model) {
      throw std::runtime_error("Failed to create model for architecture: " +
                               architecture);
    }

    model->initialize();
    std::cout << "  Model initialized successfully.\n";

    std::cout << "[3/5] Loading FP32 weights from: " << src_weight_path << "\n";
    model->load_weight(src_weight_path);
    std::cout << "  Weights loaded successfully.\n";

    std::cout << "[4/5] Quantizing and saving weights to: " << dst_weight_path
              << "\n";

    auto layer_dtype_map = buildLayerDtypeMap(num_layers, fc_dtype, embd_dtype);

    std::cout << "  Layer dtype mapping (" << layer_dtype_map.size()
              << " layers targeted):\n";
    for (const auto &[name, dt] : layer_dtype_map) {
      std::cout << "    " << name << " -> " << dataTypeToStr(dt) << "\n";
    }

    model->save_weight(dst_weight_path, DataType::NONE, layer_dtype_map);

    auto src_size = std::filesystem::file_size(src_weight_path);
    auto dst_size = std::filesystem::file_size(dst_weight_path);
    double ratio = static_cast<double>(dst_size) / src_size * 100.0;

    std::cout << "  Source size:  " << (src_size / (1024 * 1024)) << " MB\n";
    std::cout << "  Output size:  " << (dst_size / (1024 * 1024)) << " MB\n";
    std::cout << "  Compression:  " << std::fixed << std::setprecision(1)
              << ratio << "%\n";

    std::cout << "[5/5] Generating nntr_config.json...\n";

    json new_nntr_cfg = nntr_cfg;
    new_nntr_cfg["model_file_name"] = output_bin_name;
    new_nntr_cfg["fc_layer_dtype"] = dataTypeToStr(fc_dtype);
    new_nntr_cfg["embedding_dtype"] = dataTypeToStr(embd_dtype);
    new_nntr_cfg["model_tensor_type"] =
      buildModelTensorType(dataTypeToStr(fc_dtype));

    std::string output_config_path = output_dir + "/nntr_config.json";

    if (output_dir == model_path) {
      output_config_path = output_dir + "/nntr_config_quantized.json";
    }

    std::ofstream config_out(output_config_path);
    if (!config_out.is_open()) {
      throw std::runtime_error("Failed to open output config: " +
                               output_config_path);
    }
    config_out << new_nntr_cfg.dump(4) << std::endl;
    config_out.close();

    std::cout << "  Config saved to: " << output_config_path << "\n";

    std::cout << "\n";
    std::cout << "==========================================================\n";
    std::cout << "  Quantization complete!\n";
    std::cout << "==========================================================\n";
    std::cout << "\n";
    std::cout << "To run the quantized model:\n";
    if (output_dir == model_path) {
      std::cout << "  1. Rename nntr_config_quantized.json to "
                   "nntr_config.json\n";
      std::cout << "  2. nntr_xmlroberta " << model_path << "\n";
    } else {
      std::cout << "  1. Copy config.json and generation_config.json to "
                << output_dir << "\n";
      std::cout << "  2. nntr_xmlroberta " << output_dir << "\n\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
