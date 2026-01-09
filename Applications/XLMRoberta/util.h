// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   util.h
 * @date   9 January 2026
 * @brief  Utility functions for XLMRoberta
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __XLMROBERTA_UTIL_H__
#define __XLMROBERTA_UTIL_H__

#include <layer_context.h>
#include <tensor.h>

namespace xlmroberta {

/**
 * @brief Create position IDs from input IDs following the XLM-RoBERTa implementation
 * @param input_ids Input tensor containing token IDs
 * @param position_ids Output tensor to store position IDs
 * @param pad_token_id Token ID used for padding (default = 1)
 */
void createPositionIdsFromInputIds(nntrainer::Tensor &input_ids,
                                   nntrainer::Tensor &position_ids,
                                   int pad_token_id = 1);

} // namespace xlmroberta

#endif /* __XLMROBERTA_UTIL_H__ */
