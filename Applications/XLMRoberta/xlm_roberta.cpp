// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Seungback Hong <sb92.hong@samsung.com>
 * Copyright (C) 2025 Hyeonseok Lee <hs89.lee@samsung.com>
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 * Copyright (C) 2025 Junbong Yu <junbong.yu@samsung.com>
 *
 * @file   main.cpp
 * @date   06 Jan 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Junbong Yu <junbong.yu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines XLM Roberta's basic actions
 * @note   This causal_lm.h constructs a class for Transformer-based Causal
 * Language Model (CausalLM). It aims to support AutoModelForCausalLM with
 * nntrainer. It supports the following models:
 *          - intfloat/multilingual-e5-large-instruct
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include <common.h>
#include <layer_context.h>
#include <mha_core.h>
#include <embedding_layer.h>

#include <tensor.h>
#include <engine.h>
#include <app_context.h>

#include <xlm_roberta.h>
#include <llm_util.hpp>
#include <position.h>
#include <token_type.h>

using causallm::json;
using causallm::ModelType;

namespace xlmroberta
{
  static const bool debug = true; // Set to false to disable debugging

  // Helper function to extract and print input_layers parameter
  void printInputLayers(const std::vector<std::string> &params, const std::string &layer_type)
  {
    if (!debug)
      return;

    for (const auto &param : params)
    {
      if (param.find("input_layers=") == 0)
      {
        std::cout << "[DEBUG] " << layer_type << " input_layers: " << param.substr(12) << std::endl;
        break;
      }
      else if (param.find("input_layers") != std::string::npos && param.find("=") != std::string::npos)
      {
        size_t pos = param.find("=");
        std::cout << "[DEBUG] " << layer_type << " input_layers: " << param.substr(pos + 1) << std::endl;
        break;
      }
    }
  }

  XLMRoberta::XLMRoberta(causallm::json &cfg, causallm::json &generation_cfg, causallm::json &nntr_cfg)
      : causallm::Embedding(cfg, generation_cfg, nntr_cfg)
  {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  //   void XLMRoberta::initialize() {

  //   // RegisterCustomLayers
  //   XLMRoberta::registerCustomLayers();

  //   // construct causalLM model
  //   constructModel();

  //   // setup model property
  //   std::vector<std::string> model_props = {
  //     withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
  //     withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  //   if (MEMORY_SWAP) {
  //     model_props.emplace_back(withKey("fsu", "true"));
  //     model_props.emplace_back(withKey("fsu_lookahead", FSU_LOOKAHEAD));
  //   }

  //   model->setProperty(model_props);

  //   if (model->compile(ml::train::ExecutionMode::INFERENCE)) {
  //     throw std::invalid_argument("Model compilation failed.");
  //   }

  //   if (model->initialize(ml::train::ExecutionMode::INFERENCE)) {
  //     throw std::invalid_argument("Model initialization failed.");
  //   }

  //   is_initialized = true;

  // #ifdef DEBUG
  //   model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
  // #endif
  // }

  void XLMRoberta::setupParameters(json &cfg, json &generation_cfg,
                                   json &nntr_cfg)
  {
    Embedding::setupParameters(cfg, generation_cfg, nntr_cfg);

    HIDDEN_ACT = cfg["hidden_act"];

    // /** Initialize nntr prameters */
    // BATCH_SIZE = nntr_cfg["batch_size"].get<unsigned int>();
    // MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"].get<std::string>();
    // INIT_SEQ_LEN = nntr_cfg["init_seq_len"];
    // MAX_SEQ_LEN = nntr_cfg["max_seq_len"];
    // NUM_TO_GENERATE = nntr_cfg["num_to_generate"];
    // MODEL_TENSOR_TYPE = nntr_cfg["model_tensor_type"];
    // MEMORY_SWAP = nntr_cfg.contains("fsu") ? nntr_cfg["fsu"].get<bool>() : false;
    // FSU_LOOKAHEAD = nntr_cfg.contains("fsu_lookahead")
    //                     ? nntr_cfg["fsu_lookahead"].get<unsigned int>()
    //                     : 1;
    // EMBEDDING_DTYPE = nntr_cfg["embedding_dtype"];
    // FC_LAYER_DTYPE = nntr_cfg["fc_layer_dtype"];

    // /** Initialize model parameters */
    // NUM_VOCAB = cfg["vocab_size"];
    // DIM = cfg["hidden_size"];
    // INTERMEDIATE_SIZE = cfg["intermediate_size"];
    // NUM_LAYERS = cfg["num_hidden_layers"];
    // NUM_HEADS = cfg["num_attention_heads"];
    // HEAD_DIM = cfg.contains("head_dim")
    //                ? cfg["head_dim"].get<int>()
    //                : DIM / NUM_HEADS; // default value is hidden_size / num_heads
    // NUM_KEY_VALUE_HEADS = cfg.contains("num_key_value_heads")
    //                           ? cfg["num_key_value_heads"].get<int>()
    //                           : NUM_HEADS;
    // SLIDING_WINDOW =
    //     cfg.contains("sliding_window") && !cfg["sliding_window"].is_null()
    //         ? cfg["sliding_window"].get<unsigned int>()
    //         : UINT_MAX;
    // SLIDING_WINDOW_PATTERN = cfg.contains("sliding_window_pattern")
    //                              ? cfg["sliding_window_pattern"].get<unsigned int>()
    //                              : 1;
    // MAX_POSITION_EMBEDDINGS = cfg["max_position_embeddings"].get<unsigned int>();
    // // ROPE_THETA = cfg["rope_theta"].get<unsigned int>();
    // // TIE_WORD_EMBEDDINGS = cfg["tie_word_embeddings"].get<bool>();
    // // NORM_EPS = cfg["rms_norm_eps"];
    // GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

    return;
  };

  void XLMRoberta::constructXLMRobertaPooler()
  {
    std::vector<LayerHandle> layers;

    // Create output dense layer
    std::vector<std::string> output_param = {
        withKey("name", "pooler_dense"),
        withKey("unit", DIM),
        withKey("input_layers", "encoder_" + std::to_string(NUM_LAYERS - 1) + "_output_layer_norm"),
        withKey("weight_initializer", "xavier_uniform"),
        withKey("bias_initializer", "zeros")};

    printInputLayers(output_param, "pooler_dense");
    layers.push_back(createLayer("fully_connected", output_param));

    // Create output activation
    std::vector<std::string> activation_param = {
        withKey("name", "pooler_activation"),
        withKey("input_layers", "pooler_dense"),
        withKey("activation", "tanh")};

    printInputLayers(activation_param, "pooler_activation");
    layers.push_back(createLayer("activation", activation_param));

    // add created layers into the model
    for (auto &layer : layers)
    {
      model->addLayer(layer);
    }
  }

  void XLMRoberta::registerCustomLayers()
  {
    const auto &ct_engine = nntrainer::Engine::Global();
    const auto app_context =
        static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

    try
    {
      // roberta custom layers
      app_context->registerFactory(nntrainer::createLayer<xlmroberta::Position>);
      app_context->registerFactory(nntrainer::createLayer<xlmroberta::TokenType>);

      // causallm custom layers
      app_context->registerFactory(nntrainer::createLayer<causallm::EmbeddingLayer>);
      app_context->registerFactory(nntrainer::createLayer<causallm::MHACoreLayer>);
    }
    catch (std::invalid_argument &e)
    {
      std::cerr << "failed to register factory, reason: " << e.what()
                << std::endl;
    }
  }

  void XLMRoberta::constructXLMRobertaOutput(int layer_id, std::string input_name)
  {
    std::vector<LayerHandle> layers;

    // Create output dense layer
    std::vector<std::string> output_param = {
        withKey("name", "encoder_" + std::to_string(layer_id) + "_output_dense"),
        withKey("unit", DIM),
        withKey("input_layers", input_name),
        withKey("weight_initializer", "xavier_uniform"),
        withKey("bias_initializer", "zeros")};

    printInputLayers(output_param, "encoder_" + std::to_string(layer_id) + "_output_dense");
    layers.push_back(createLayer("fully_connected", output_param));

    // Add layer normalization
    std::vector<std::string> norm_param = {
        withKey("name", "encoder_" + std::to_string(layer_id) + "_output_layer_norm"),
        withKey("axis", "3"),
        withKey("epsilon", "1e-5"),
        withKey("input_layers", "encoder_" + std::to_string(layer_id) + "_output_dense")};
    printInputLayers(norm_param, "encoder_" + std::to_string(layer_id) + "_output_layer_norm");
    layers.push_back(createLayer("layer_normalization", norm_param));

    // add created layers into the model
    for (auto &layer : layers)
    {
      model->addLayer(layer);
    }
  }

  void XLMRoberta::constructXLMRobertaIntermediate(int layer_id, std::string input_name)
  {
    std::vector<LayerHandle> layers;

    // Create intermediate dense layer
    std::vector<std::string> intermediate_param = {
        withKey("name", "encoder_" + std::to_string(layer_id) + "_intermediate_dense"),
        withKey("unit", INTERMEDIATE_SIZE),
        withKey("input_layers", input_name),
        withKey("weight_initializer", "xavier_uniform"),
        withKey("bias_initializer", "zeros")};
    printInputLayers(intermediate_param, "encoder_" + std::to_string(layer_id) + "_intermediate_dense");
    layers.push_back(createLayer("fully_connected", intermediate_param));

    // Add activation layer (GELU)
    std::vector<std::string> activation_param = {
        withKey("name", "encoder_" + std::to_string(layer_id) + "_intermediate_act"),
        withKey("input_layers", "encoder_" + std::to_string(layer_id) + "_intermediate_dense"),
        withKey("activation", "gelu")};
    printInputLayers(activation_param, "encoder_" + std::to_string(layer_id) + "_intermediate_act");
    layers.push_back(createLayer("activation", activation_param));

    // add created layers into the model
    for (auto &layer : layers)
    {
      model->addLayer(layer);
    }
  }

  void XLMRoberta::constructXLMRobertaSelfOutput(int layer_id, std::string input_name)
  {
    std::vector<LayerHandle> layers;

    std::vector<std::string> output_param = {
        withKey("name", "encoder_" + std::to_string(layer_id) + "_attention_output_dense"),
        withKey("unit", DIM),
        withKey("disable_bias", "false"),
        withKey("input_layers", input_name),
        withKey("weight_initializer", "ones")};

    printInputLayers(output_param, "encoder_" + std::to_string(layer_id) + "_attention_output_dense");
    layers.push_back(createLayer("fully_connected", output_param));

    std::vector<std::string> layer_norm_params = {
        withKey("name", "encoder_" + std::to_string(layer_id) + "_attention_output_layer_norm"),
        withKey("axis", "3"),
        withKey("epsilon", "1e-5"),
        withKey("input_layers", "encoder_" + std::to_string(layer_id) + "_attention_output_dense")};

    printInputLayers(layer_norm_params, "encoder_" + std::to_string(layer_id) + "_attention_output_layer_norm");
    layers.push_back(createLayer("layer_normalization", layer_norm_params));

    // add created layers into the model
    for (auto &layer : layers)
    {
      model->addLayer(layer);
    }
  }

  std::vector<LayerHandle>
  XLMRoberta::createAttention(const int layer_id, int seq_len, int n_heads,
                              int head_dim, std::string query_name,
                              std::string key_name, std::string value_name)
  {

    std::vector<LayerHandle> layers;

    auto Q = "encoder_" + std::to_string(layer_id) + "_attention_self_wq";
    auto K = "encoder_" + std::to_string(layer_id) + "_attention_self_wk";
    auto V = "encoder_" + std::to_string(layer_id) + "_attention_self_wv";
    auto A = "encoder_" + std::to_string(layer_id) + "_attention_self_attention";

    // V layer
    std::vector<std::string> v_params = {
        withKey("name", V), withKey("unit", head_dim * n_heads),
        withKey("disable_bias", "false"), withKey("input_layers", value_name),
        withKey("weight_initializer", "ones")};
    printInputLayers(v_params, V);
    layers.push_back(createLayer("fully_connected", v_params));

    // K layer
    std::vector<std::string> k_params = {
        withKey("name", K), withKey("unit", head_dim * n_heads),
        withKey("disable_bias", "false"), withKey("input_layers", key_name),
        withKey("weight_initializer", "ones")};
    printInputLayers(k_params, K);
    layers.push_back(createLayer("fully_connected", k_params));

    // Q layer
    std::vector<std::string> q_params = {
        withKey("name", Q), withKey("unit", head_dim * n_heads),
        withKey("disable_bias", "false"), withKey("input_layers", query_name),
        withKey("weight_initializer", "ones")};
    printInputLayers(q_params, Q);
    layers.push_back(createLayer("fully_connected", q_params));

    // Attention core layer
    std::vector<std::string> a_params = {
        withKey("name", A),
        withKey("num_heads", n_heads),
        withKey("num_heads_KV", n_heads),
        withKey("max_timestep", std::to_string(MAX_POSITION_EMBEDDINGS)), // (JBD) is this correct?
        withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
        withKey("input_layers", {Q, K, V})};
    printInputLayers(a_params, A);
    layers.push_back(createLayer("mha_core", a_params));

    return layers;
  }

  void XLMRoberta::constructXLMRobertaSelfAttention(int layer_id, std::string input_name)
  {
    std::vector<LayerHandle> layers;

    // create transformer layers

    std::vector<LayerHandle> self_attn;

    self_attn = createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM,
                                input_name,  // Use the input_name for query
                                input_name,  // Use the input_name for key
                                input_name); // Use the input_name for value

    layers.insert(layers.end(), self_attn.begin(), self_attn.end());

    // add created layers into the model
    for (auto &layer : layers)
    {
      model->addLayer(layer);
    }
  }

  void XLMRoberta::constructXLMRobertaAttention(int layer_id, std::string input_name)
  {
    constructXLMRobertaSelfAttention(layer_id, input_name);
    input_name = "encoder_" + std::to_string(layer_id) + "_attention_self_attention";
    constructXLMRobertaSelfOutput(layer_id, input_name);
  }

  void XLMRoberta::constructXLMRobertaLayer()
  {
    std::string input_name = "embedding_layer_norm"; // Start with embedding output

    for (int i = 0; i < NUM_LAYERS; ++i)
    {
      // Construct attention sub-layer
      constructXLMRobertaAttention(i, input_name);

      // Update input_name to the output of attention
      input_name = "encoder_" + std::to_string(i) + "_attention_output_layer_norm";

      // Construct intermediate sub-layer
      constructXLMRobertaIntermediate(i, input_name);

      // Update input_name to the output of intermediate
      input_name = "encoder_" + std::to_string(i) + "_intermediate_act";

      // Construct output sub-layer
      constructXLMRobertaOutput(i, input_name);

      // Update input_name to the final output of this layer
      input_name = "encoder_" + std::to_string(i) + "_output_layer_norm";
    }
  }

  void XLMRoberta::constructXLMRobertaEncoder()
  {
    constructXLMRobertaLayer();
  }

  void XLMRoberta::constructXLMRobertaEmbeddings()
  {
    std::vector<LayerHandle> layers;

    // create input layer
    std::vector<std::string> input_params = {withKey("name", "input0"),
                                             withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))};
    printInputLayers(input_params, "input");
    layers.push_back(createLayer("input", input_params));

    // create embedding layer
    const std::string embedding_type =
        TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

    // Custom: position_ids
    std::vector<std::string> position_params = {
        "name=position",
        "weight_dtype=" + EMBEDDING_DTYPE,
        "out_dim=" + std::to_string(DIM),
        "input_layers=input0"};
    printInputLayers(position_params, "position");
    layers.push_back(createLayer("position", position_params));

    // Custom: token_type_ids
    std::vector<std::string> token_type_params = {
        "name=token_type",
        "weight_dtype=" + EMBEDDING_DTYPE,
        "out_dim=" + std::to_string(DIM),
        "input_layers=input0"};
    printInputLayers(token_type_params, "token_type");
    layers.push_back(createLayer("token_type", token_type_params));

    // word_embedding
    std::vector<std::string> word_embedding_params = {
        "name=embedding_word_embedding",
        "in_dim=" + std::to_string(NUM_VOCAB),
        "weight_dtype=" + EMBEDDING_DTYPE,
        "out_dim=" + std::to_string(DIM),
        "input_layers=input0"};
    printInputLayers(word_embedding_params, "embedding_word_embedding");
    layers.push_back(createLayer(embedding_type, word_embedding_params));

    // position_embedding
    std::vector<std::string> position_embedding_params = {
        "name=embedding_position_embedding",
        "in_dim=" + std::to_string(NUM_VOCAB),
        "weight_dtype=" + EMBEDDING_DTYPE,
        "out_dim=" + std::to_string(DIM),
        "input_layers=position"};
    printInputLayers(position_embedding_params, "embedding_position_embedding");
    layers.push_back(createLayer(embedding_type, position_embedding_params));

    // token_type_embedding
    std::vector<std::string> token_type_embedding_params = {
        "name=embedding_token_type_embedding",
        "in_dim=" + std::to_string(NUM_VOCAB),
        "weight_dtype=" + EMBEDDING_DTYPE,
        "out_dim=" + std::to_string(DIM),
        "input_layers=token_type"};
    printInputLayers(token_type_embedding_params, "embedding_token_type_embedding");
    layers.push_back(createLayer(embedding_type, token_type_embedding_params));

    // addition0: embeddings = inputs_embeds + token_type_embeddings
    std::vector<std::string> addition0_params = {withKey("name", "embedding_addition0"),
                                                 withKey("input_layers", "embedding_word_embedding,embedding_token_type_embedding")};
    printInputLayers(addition0_params, "embedding_addition0");
    layers.push_back(createLayer("addition", addition0_params));

    // addition1: embeddings + position_embeddings
    std::vector<std::string> addition1_params = {
        withKey("name", "embedding_addition1"),
        withKey("input_layers", "embedding_addition0,embedding_position_embedding")};
    printInputLayers(addition1_params, "embedding_addition1");
    layers.push_back(createLayer("addition", addition1_params));

    // layer normalization
    std::vector<std::string> layer_norm_params = {"name=embedding_layer_norm",
                                                  "axis=3",
                                                  "epsilon=1e-5",
                                                  "input_layers=embedding_addition1"};
    printInputLayers(layer_norm_params, "embedding_layer_norm");
    layers.push_back(createLayer("layer_normalization", layer_norm_params));

    for (auto &layer : layers)
    {
      model->addLayer(layer);
    }
  }

  void XLMRoberta::constructModel()
  {
    // create model
    model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

    for (auto &module : modules)
    {
      if (!module.contains("type"))
      {
        continue;
      }
      std::string type = module["type"].get<std::string>();
      std::string component = getLastComponent(type);

      if (component == "Transformer")
      {
        constructXLMRobertaEmbeddings();
        constructXLMRobertaEncoder();
        constructXLMRobertaPooler();
      }
      else
      {
        if (module.contains("idx"))
        {
          int idx = module["idx"].get<int>();
          // Add module layer using properties from loaded config
          // addModule(type, idx);
        }
        else
        {
          std::cerr << "Warning: Module does not have idx field, skipping: "
                    << type << std::endl;
        }
      }
    }
  }

} // namespace xlmroberta
