// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   position.cpp
 * @date   7 January 2026
 * @brief  Implementation of position layer for XLMRoberta
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "position.h"
#include "util.h"

namespace xlmroberta {

static constexpr size_t SINGLE_INOUT_IDX = 0;
static constexpr int PAD_TOKEN_ID = 1;

void Position::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions(context.getInputDimensions());
}

void Position::calcDerivative(nntrainer::RunLayerContext &context) {
  // std::throw_with_nested(std::runtime_error(
  //     "Training is not supported yet."));
}

void Position::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  // Get input tensor (input_ids)
  nntrainer::Tensor &input_ids = context.getInput(SINGLE_INOUT_IDX);

  // Get output tensor (position_ids)
  nntrainer::Tensor &position_ids = context.getOutput(SINGLE_INOUT_IDX);

  // Create position IDs using utility function
  // pad_token_id is 1 (hardcoded as requested)
  createPositionIdsFromInputIds(input_ids, position_ids, PAD_TOKEN_ID);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_position_layer() {
  auto layer = new Position();
  std::cout << "position layer created\n";
  return layer;
}

void destroy_position_layer(nntrainer::Layer *layer) {
  std::cout << "position layer deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_position_layer,
                                                   destroy_position_layer};
}

#endif

} // namespace xlmroberta
