// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   inference_dispatcher.cpp
 * @brief  InferenceDispatcher implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "inference_dispatcher.h"
#include "causal_lm.h"
#include "model_pool.h"
#include "transformer.h"

#include <iostream>

namespace causallm {
namespace api {

static std::string apply_chat_template_internal(const std::string &architecture,
                                                const std::string &input) {
  if (architecture == "LlamaForCausalLM") {
    return "[INST] " + input + " [/INST]";
  } else if (architecture == "Qwen2ForCausalLM" ||
             architecture == "Qwen3ForCausalLM" ||
             architecture == "Qwen3MoeForCausalLM" ||
             architecture == "Qwen3SlimMoeForCausalLM" ||
             architecture == "Qwen3CachedSlimMoeForCausalLM") {
    return "<|im_start|>user\n" + input + "<|im_end|>\n<|im_start|>assistant\n";
  } else if (architecture == "Gemma3ForCausalLM") {
    return "<start_of_turn>user\n" + input +
           "<end_of_turn>\n<start_of_turn>model\n";
  }
  return input;
}

InferenceDispatcher &InferenceDispatcher::Instance() {
  static InferenceDispatcher dispatcher;
  return dispatcher;
}

InferenceDispatcher::InferenceDispatcher() :
  thread_count_(std::thread::hardware_concurrency()) {}

std::future<InferenceResult>
InferenceDispatcher::submit(const std::string &model_key,
                            const std::string &prompt, bool use_chat_template,
                            bool verbose, const std::string &architecture) {

  int request_id = next_request_id_++;

  return std::async(std::launch::async, [model_key, prompt, use_chat_template,
                                         verbose, architecture,
                                         request_id]() -> InferenceResult {
    auto session = ModelPool::Instance().createSession(model_key);
    if (!session || !session->model) {
      InferenceResult result;
      result.error_code = CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
      return result;
    }

    session->is_running = true;

    InferenceResult result;
    try {
      std::string input = prompt;
      if (use_chat_template) {
        input = apply_chat_template_internal(architecture, input);
      }

      session->model->run(input, false, "", "", verbose);

      auto causal_lm = dynamic_cast<causallm::CausalLM *>(session->model.get());
      if (causal_lm) {
        result.output = causal_lm->getOutput(0);
        auto m = causal_lm->getPerformanceMetrics();
        result.metrics.prefill_tokens = m.prefill_tokens;
        result.metrics.prefill_duration_ms = m.prefill_duration_ms;
        result.metrics.generation_tokens = m.generation_tokens;
        result.metrics.generation_duration_ms = m.generation_duration_ms;
        result.metrics.total_duration_ms = m.total_duration_ms;
        result.metrics.initialization_duration_ms = m.initialization_duration_ms;
        result.metrics.peak_memory_kb = m.peak_memory_kb;
      }

      result.error_code = CAUSAL_LM_ERROR_NONE;
    } catch (const std::exception &e) {
      std::cerr << "[InferenceDispatcher] Request " << request_id
                << " failed: " << e.what() << std::endl;
      result.error_code = CAUSAL_LM_ERROR_INFERENCE_FAILED;
    }

    session->is_running = false;
    ModelPool::Instance().releaseSession(session.get());

    return result;
  });
}

InferenceResult InferenceDispatcher::runSync(const std::string &model_key,
                                             const std::string &prompt,
                                             bool use_chat_template,
                                             bool verbose,
                                             const std::string &architecture) {
  auto future =
    submit(model_key, prompt, use_chat_template, verbose, architecture);
  return future.get();
}

void InferenceDispatcher::setThreadCount(size_t count) { thread_count_ = count; }

size_t InferenceDispatcher::getThreadCount() const {
  return thread_count_;
}

} // namespace api
} // namespace causallm
