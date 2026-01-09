// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   token_type.cpp
 * @date   7 January 2026
 * @brief  Implementation of token type layer for XLMRoberta
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "token_type.h"
#include "util.h"

namespace xlmroberta {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void TokenType::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions(context.getInputDimensions());
}

void TokenType::calcDerivative(nntrainer::RunLayerContext &context) {
  // std::throw_with_nested(std::runtime_error("Training is not supported
  // yet."));
}

void TokenType::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  // Get input tensor (input_ids)
  nntrainer::Tensor &input_ids = context.getInput(SINGLE_INOUT_IDX);

  // Get output tensor (token_type_ids)
  nntrainer::Tensor &token_type_ids = context.getOutput(SINGLE_INOUT_IDX);

  // For XLM-RoBERTa, token_type_ids are typically all zeros since it processes
  // single sentences. We follow the same approach as the Python implementation
  // by creating position_ids first and then setting all token_type_ids to 0.

  // Create temporary position_ids tensor with same dimensions as input
  nntrainer::Tensor position_ids(input_ids.getDim());

  // Generate position_ids from input_ids
  createPositionIdsFromInputIds(input_ids, position_ids, 1);

  // Set all token_type_ids to 0 (following XLM-RoBERTa convention for single sentences)
  token_type_ids.setValue(0);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_token_type_layer() {
  auto layer = new TokenType();
  std::cout << "token type layer created\n";
  return layer;
}

void destroy_token_type_layer(nntrainer::Layer *layer) {
  std::cout << "token type layer deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_token_type_layer,
                                                   destroy_token_type_layer};
}

#endif

} // namespace xlmroberta
