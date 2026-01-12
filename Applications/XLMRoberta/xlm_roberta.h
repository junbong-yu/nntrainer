// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file    embedding.h
 * @date    02 Jan 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 * @note    This embedding.h constructs a class for XLMRoberta model
 * which can be a parent of models with embedding (encoder) structure.
 */

#ifndef __XLM_ROBERTA_H__
#define __XLM_ROBERTA_H__

#pragma once

#include <map>
#include "../CausalLM/models/embedding.h"

namespace xlmroberta
{
  /**
   * @brief XLMRoberta Class
   */
  WIN_EXPORT class XLMRoberta : public causallm::Embedding
  {

  public:
    // void initialize();
    static constexpr const char *architectures = "XLMRobertaModel";
    std::string HIDDEN_ACT;

    void registerCustomLayers();

    void setupParameters(causallm::json &cfg, causallm::json &generation_cfg, causallm::json &nntr_cfg);

    /**
     * @brief Construct a new XLMRoberta object
     * @param cfg Configuration for the model (config.json)
     * @param generation_cfg Configuration for the generation (generation.json)
     * @param nntr_cfg Configuration for nntrainer (nntr_config.json)
     */
    XLMRoberta(causallm::json &cfg, causallm::json &generation_cfg, causallm::json &nntr_cfg);

    /**
     * @brief Destroy the XLMRoberta object
     */
    virtual ~XLMRoberta() {}

  protected:
    /**
     * @brief Construct Model
     */
    void constructModel();

  private:
    void constructXLMRobertaEmbeddings();

    void constructXLMRobertaEncoder();
    void constructXLMRobertaLayer();                                             // x24 times
    void constructXLMRobertaAttention(int layer_id, std::string input_name);     // #1
    void constructXLMRobertaSelfAttention(int layer_id, std::string input_name); // #1-1
    void constructXLMRobertaSelfOutput(int layer_id, std::string input_name);    // #1-2
    void constructXLMRobertaIntermediate(int layer_id, std::string input_name);  // #2
    void constructXLMRobertaOutput(int layer_id, std::string input_name);        // #3

    void constructXLMRobertaPooler();
  };
} // namespace xlmroberta

#endif
