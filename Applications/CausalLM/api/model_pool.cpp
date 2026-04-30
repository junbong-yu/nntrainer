// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   model_pool.cpp
 * @brief  ModelPool implementation for multi-session inference
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "model_pool.h"
#include "factory.h"
#include "transformer.h"

#include <iostream>

namespace causallm {
namespace api {

ModelPool &ModelPool::Instance() {
  static ModelPool pool;
  return pool;
}

bool ModelPool::loadModel(const std::string &model_key,
                          const std::string &architecture, nlohmann::json cfg,
                          nlohmann::json generation_cfg,
                          nlohmann::json nntr_cfg,
                          const std::string &weight_path) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (pool_.find(model_key) != pool_.end()) {
    return true;
  }

  PooledModel pm;
  pm.model_key = model_key;
  pm.architecture = architecture;
  pm.cfg = std::move(cfg);
  pm.generation_cfg = std::move(generation_cfg);
  pm.nntr_cfg = std::move(nntr_cfg);
  pm.weight_path = weight_path;
  pm.ref_count = 0;

  pool_[model_key] = std::move(pm);
  return true;
}

std::unique_ptr<InferenceSession>
ModelPool::createSession(const std::string &model_key) {

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = pool_.find(model_key);
  if (it == pool_.end()) {
    std::cerr << "[ModelPool] Model not loaded: " << model_key << std::endl;
    return nullptr;
  }

  auto &pm = it->second;

  auto model = Factory::Instance().create(pm.architecture, pm.cfg,
                                          pm.generation_cfg, pm.nntr_cfg);
  if (!model) {
    std::cerr << "[ModelPool] Failed to create model instance for: "
              << model_key << std::endl;
    return nullptr;
  }

  try {
    model->initialize();
    model->load_weight(pm.weight_path);
  } catch (const std::exception &e) {
    std::cerr << "[ModelPool] Failed to initialize/load model: " << e.what()
              << std::endl;
    return nullptr;
  }

  auto session = std::make_unique<InferenceSession>();
  session->id = next_session_id_++;
  session->model_key = model_key;
  session->model = std::move(model);
  session->is_initialized = true;
  session->created_at = std::chrono::steady_clock::now();

  pm.ref_count++;

  return session;
}

void ModelPool::releaseSession(InferenceSession *session) {
  if (!session)
    return;

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = pool_.find(session->model_key);
  if (it != pool_.end()) {
    it->second.ref_count--;
    if (it->second.ref_count < 0) {
      it->second.ref_count = 0;
    }
  }
}

bool ModelPool::isLoaded(const std::string &model_key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pool_.find(model_key) != pool_.end();
}

bool ModelPool::unloadModel(const std::string &model_key) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = pool_.find(model_key);
  if (it == pool_.end()) {
    return false;
  }

  if (it->second.ref_count > 0) {
    std::cerr << "[ModelPool] Cannot unload model " << model_key
              << ": still has " << it->second.ref_count << " active session(s)"
              << std::endl;
    return false;
  }

  pool_.erase(it);
  return true;
}

} // namespace api
} // namespace causallm
