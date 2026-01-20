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
  WIN_EXPORT class XLMRoberta : public causallm::Embedding
  {

  public:
  // void initialize();
    static constexpr const char *architectures = "XLMRobertaModel";
    std::string HIDDEN_ACT;
    XLMRoberta(
        causallm::json &cfg, causallm::json &generation_cfg, causallm::json &nntr_cfg)
        : Transformer(cfg, generation_cfg, nntr_cfg, causallm::ModelType::EMBEDDING),
          Embedding(cfg, generation_cfg, nntr_cfg)
    {
      setupParameters(cfg, generation_cfg, nntr_cfg);
    }

    virtual ~XLMRoberta() = default;

  private:
    
    void constructModel();
    void setupParameters(causallm::json &cfg, causallm::json &generation_cfg,
                         causallm::json &nntr_cfg);
    void constructXLMRobertaPooler();
    void constructXLMRobertaEncoder();
    void constructXLMRobertaEmbeddings();
    void constructXLMRobertaModel();
    void registerCustomLayers();
    void constructXLMRobertaOutput(int, std::string);
    void constructXLMRobertaIntermediate(int, std::string);
    void constructXLMRobertaSelfOutput(int, std::string);
    std::vector<std::shared_ptr<ml::train::Layer>> createAttention(int, int, int, int,
       std::string, std::string, std::string);
    void constructXLMRobertaSelfAttention(int, std::string);
    void constructXLMRobertaAttention(int, std::string);
    void constructXLMRobertaLayer();
    std::string getLastComponent(const std::string &type);

    std::vector<causallm::json> modules;
  };

} // namespace xlmroberta

#endif
