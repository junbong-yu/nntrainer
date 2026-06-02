// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   inference_dispatcher.h
 * @brief  Async inference dispatcher with thread pool
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __INFERENCE_DISPATCHER_H__
#define __INFERENCE_DISPATCHER_H__

#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include "causal_lm_api.h"
#include "performance_metrics.h"

namespace causallm {
namespace api {

/**
 * @brief Result of an async inference request
 */
struct InferenceResult {
  std::string output;         /**< Generated output text */
  PerformanceMetrics metrics; /**< Performance metrics */
  ErrorCode error_code;       /**< Error code */
};

/**
 * @brief Dispatches inference requests to a thread pool
 *
 * Each request creates an independent session, runs inference asynchronously,
 * and returns the result via std::future.
 */
class InferenceDispatcher {
public:
  /**
   * @brief Get the process-wide InferenceDispatcher singleton
   */
  static InferenceDispatcher &Instance();

  /**
   * @brief Submit an async inference request
   *
   * @param model_key Model key in ModelPool
   * @param prompt Input prompt text
   * @param use_chat_template Whether to apply chat template
   * @param verbose Whether to print output during generation
   * @param architecture Model architecture name (for chat template)
   * @return std::future<InferenceResult> Future holding the result
   */
  std::future<InferenceResult> submit(const std::string &model_key,
                                      const std::string &prompt,
                                      bool use_chat_template, bool verbose,
                                      const std::string &architecture);

  /**
   * @brief Run inference synchronously
   *
   * @param model_key Model key in ModelPool
   * @param prompt Input prompt text
   * @param use_chat_template Whether to apply chat template
   * @param verbose Whether to print output during generation
   * @param architecture Model architecture name (for chat template)
   * @return InferenceResult Result of the inference
   */
  InferenceResult runSync(const std::string &model_key,
                          const std::string &prompt, bool use_chat_template,
                          bool verbose, const std::string &architecture);

  /**
   * @brief Set the number of threads in the pool
   */
  void setThreadCount(size_t count);

  /**
   * @brief Get the current number of threads
   */
  size_t getThreadCount() const;

private:
  InferenceDispatcher();
  ~InferenceDispatcher() = default;

  std::atomic<int> next_request_id_{1};
  size_t thread_count_;
};

} // namespace api
} // namespace causallm

#endif // __INFERENCE_DISPATCHER_H__
