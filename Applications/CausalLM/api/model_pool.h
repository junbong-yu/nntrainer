// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   model_pool.h
 * @brief  Process-wide model pool for multi-session inference
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __MODEL_POOL_H__
#define __MODEL_POOL_H__

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "inference_session.h"
#include "json.hpp"

namespace causallm {
namespace api {

/**
 * @brief Manages loaded model templates and creates inference sessions
 *
 * ModelPool is a process-wide singleton that maintains a pool of loaded
 * model templates. Each template stores the configuration needed to create
 * independent model instances that share weights via OS page cache.
 */
class ModelPool {
public:
  /**
   * @brief Model template stored in the pool
   */
  struct PooledModel {
    std::string model_key;
    std::string architecture;
    nlohmann::json cfg;
    nlohmann::json generation_cfg;
    nlohmann::json nntr_cfg;
    std::string weight_path;
    int ref_count = 0;
  };

  /**
   * @brief Get the process-wide ModelPool singleton
   */
  static ModelPool &Instance();

  /**
   * @brief Load a model template into the pool
   *
   * @param model_key Unique model key (e.g., "QWEN3-0.6B-W16A16")
   * @param architecture Architecture name for Factory
   * @param cfg Model configuration JSON
   * @param generation_cfg Generation configuration JSON
   * @param nntr_cfg NNTrainer configuration JSON
   * @param weight_path Path to the weight binary file
   * @return true if loaded successfully (or already loaded)
   * @return false on error
   */
  bool loadModel(const std::string &model_key, const std::string &architecture,
                 nlohmann::json cfg, nlohmann::json generation_cfg,
                 nlohmann::json nntr_cfg, const std::string &weight_path);

  /**
   * @brief Create a new inference session for a loaded model
   *
   * Creates an independent Transformer instance from the template.
   * Weights are loaded via load_weight(), leveraging OS page cache
   * for sharing across sessions of the same model.
   *
   * @param model_key Model key in the pool
   * @return Unique pointer to InferenceSession, or nullptr on error
   */
  std::unique_ptr<InferenceSession> createSession(const std::string &model_key);

  /**
   * @brief Release a session back to the pool
   *
   * Decrements the reference count. The session object is destroyed.
   *
   * @param session Pointer to the session being released
   */
  void releaseSession(InferenceSession *session);

  /**
   * @brief Check if a model template is loaded in the pool
   */
  bool isLoaded(const std::string &model_key) const;

  /**
   * @brief Unload a model template from the pool
   *
   * Only succeeds if ref_count is 0.
   *
   * @param model_key Model key to unload
   * @return true if unloaded, false if not found or still in use
   */
  bool unloadModel(const std::string &model_key);

private:
  ModelPool() = default;
  ~ModelPool() = default;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, PooledModel> pool_;
  int next_session_id_ = 1;
};

} // namespace api
} // namespace causallm

#endif // __MODEL_POOL_H__
