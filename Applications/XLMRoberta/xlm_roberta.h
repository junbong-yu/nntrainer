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

    void constructXLMRobertaModel();
  };

} // namespace xlmroberta

#endif
