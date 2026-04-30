// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   inference_session.h
 * @brief  Session structure for independent inference execution
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __INFERENCE_SESSION_H__
#define __INFERENCE_SESSION_H__

#include <atomic>
#include <chrono>
#include <memory>
#include <string>

namespace causallm {
class Transformer;
}

namespace causallm {
namespace api {

/**
 * @brief Represents a single inference session with independent state
 *
 * Each session holds its own Transformer model instance, ensuring
 * thread-safe concurrent inference across multiple sessions.
 */
struct InferenceSession {
  int id;                /**< Unique session identifier */
  std::string model_key; /**< Model key in the pool */
  std::unique_ptr<causallm::Transformer>
    model;                     /**< Independent model instance */
  bool is_initialized = false; /**< Whether the session is ready */

  /** Session creation timestamp */
  std::chrono::steady_clock::time_point created_at;

  /** True if the session is currently running inference */
  std::atomic<bool> is_running{false};

  /** Per-session config overrides */
  bool use_chat_template = false;
  bool verbose = false;

  InferenceSession() = default;
  ~InferenceSession() = default;

  // Disable copy to prevent accidental sharing of model instance
  InferenceSession(const InferenceSession &) = delete;
  InferenceSession &operator=(const InferenceSession &) = delete;

  // Enable move
  InferenceSession(InferenceSession &&) = default;
  InferenceSession &operator=(InferenceSession &&) = default;
};

} // namespace api
} // namespace causallm

#endif // __INFERENCE_SESSION_H__
