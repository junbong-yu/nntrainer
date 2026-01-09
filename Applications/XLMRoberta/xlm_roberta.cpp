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

  XLMRoberta::XLMRoberta(causallm::json &cfg, causallm::json &generation_cfg, causallm::json &nntr_cfg)
      : causallm::Embedding(cfg, generation_cfg, nntr_cfg)
  {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  void XLMRoberta::constructModel()
  {
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
        constructXLMRobertaModel();
      }
      else
      {
        if (module.contains("idx"))
        {
          int idx = module["idx"].get<int>();
          // Add module layer using properties from loaded config
          addModule(type, idx);
        }
        else
        {
          std::cerr << "Warning: Module does not have idx field, skipping: "
                    << type << std::endl;
        }
      }
    }
  }

  void XLMRoberta::constructXLMRobertaModel()
  {

    std::vector<LayerHandle> layers;

    // create model
    model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

    // create input layer
    layers.push_back(createLayer(
        "input", {withKey("name", "input0"),
                  withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

    // create embedding layer
    const std::string embedding_type =
        TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

    // Custom: position_ids
    layers.push_back(createLayer(
        "position",
        {"name=position",
         "weight_dtype=" + EMBEDDING_DTYPE,
         "out_dim=" + std::to_string(DIM),
         "input_layer=input0"}));

    // Custom: token_type_ids
    layers.push_back(createLayer(
        "token_type",
        {"name=token_type",
         "weight_dtype=" + EMBEDDING_DTYPE,
         "out_dim=" + std::to_string(DIM),
         "input_layer=input0"}));

    // embedding0: word_embedding
    layers.push_back(createLayer(
        embedding_type,
        {"name=embedding0_word_embedding",
         "in_dim=" + std::to_string(NUM_VOCAB),
         "weight_dtype=" + EMBEDDING_DTYPE,
         "out_dim=" + std::to_string(DIM),
         "input_layer=input0"}));

    // embedding0: position_embedding
    layers.push_back(createLayer(
        embedding_type,
        {"name=embedding0_position_embedding",
         "in_dim=" + std::to_string(NUM_VOCAB),
         "weight_dtype=" + EMBEDDING_DTYPE,
         "out_dim=" + std::to_string(DIM),
         "input_layer=position"}));

    // embedding0: token_type_embedding
    layers.push_back(createLayer(
        embedding_type,
        {"name=embedding0_token_type_embedding",
         "in_dim=" + std::to_string(NUM_VOCAB),
         "weight_dtype=" + EMBEDDING_DTYPE,
         "out_dim=" + std::to_string(DIM),
         "input_layer=token_type"}));

    // create transformer layers
    for (int i = 0; i < NUM_LAYERS; ++i)
    {
      std::vector<LayerHandle> transformer;
      if (i == 0)
        transformer = createTransformerDecoderBlock(0, "embedding0");
      else
        transformer = createTransformerDecoderBlock(
            i, "layer" + std::to_string(i - 1) + "_decoder_output");
      layers.insert(layers.end(), transformer.begin(), transformer.end());
    }

    // create rms_norm
    layers.push_back(createLayer(
        "rms_norm",
        {withKey("name", "output_norm"),
         withKey("epsilon", std::to_string(NORM_EPS)),
         withKey("input_layers",
                 "layer" + std::to_string(NUM_LAYERS - 1) + "_decoder_output"),
         withKey("packed", "false")}));

    // add created layers into the model
    for (auto &layer : layers)
    {
      model->addLayer(layer);
    }
  }

} // namespace xlmroberta
