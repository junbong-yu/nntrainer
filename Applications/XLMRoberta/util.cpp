// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   util.cpp
 * @date   9 January 2026
 * @brief  Utility functions for XLMRoberta
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "util.h"

namespace xlmroberta {

void createPositionIdsFromInputIds(nntrainer::Tensor &input_ids,
                                   nntrainer::Tensor &position_ids,
                                   int pad_token_id) {
  // Get tensor dimensions
  auto input_dim = input_ids.getDim();
  unsigned int batch_size = input_dim.batch();
  unsigned int seq_length = input_dim.width();

  // Create position_ids from input_ids following the Python implementation:
  // 1. Create mask where non-padding tokens are 1, padding tokens are 0
  // 2. Calculate cumulative sum to get sequential position numbers
  // 3. Apply offset to start from pad_token_id + 1
  for (unsigned int b = 0; b < batch_size; ++b) {
    int cumulative_count = 0;
    for (unsigned int i = 0; i < seq_length; ++i) {
      int token_id = static_cast<int>(input_ids.getValue(b, 0, 0, i));
      
      if (token_id != pad_token_id) {
        // Non-padding token gets incremental position starting from pad_token_id + 1
        cumulative_count++;
        position_ids.setValue(b, 0, 0, i, static_cast<float>(cumulative_count + pad_token_id));
      } else {
        // Padding token gets pad_token_id
        position_ids.setValue(b, 0, 0, i, static_cast<float>(pad_token_id));
      }
    }
  }
}

} // namespace xlmroberta
