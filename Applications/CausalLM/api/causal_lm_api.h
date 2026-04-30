// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    causal_lm_api.h
 * @date    21 Jan 2026
 * @brief   This is a C API for CausalLM application
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#ifndef __CAUSAL_LM_API_H__
#define __CAUSAL_LM_API_H__

#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief Performance Metrics
 */
typedef struct {
  unsigned int prefill_tokens;
  double prefill_duration_ms;
  unsigned int generation_tokens;
  double generation_duration_ms;
  double total_duration_ms;
  double initialization_duration_ms;
  size_t peak_memory_kb;
} PerformanceMetrics;

/**
 * @brief Error codes
 */
typedef enum {
  CAUSAL_LM_ERROR_NONE = 0,
  CAUSAL_LM_ERROR_INVALID_PARAMETER = 1,
  CAUSAL_LM_ERROR_MODEL_LOAD_FAILED = 2,
  CAUSAL_LM_ERROR_INFERENCE_FAILED = 3,
  CAUSAL_LM_ERROR_NOT_INITIALIZED = 4,
  CAUSAL_LM_ERROR_INFERENCE_NOT_RUN = 5,
  CAUSAL_LM_ERROR_UNKNOWN = 99
} ErrorCode;

/**
 * @brief Backend compute type
 */
typedef enum {
  CAUSAL_LM_BACKEND_CPU = 0,
  CAUSAL_LM_BACKEND_GPU = 1, /// < @todo: support gpu
  CAUSAL_LM_BACKEND_NPU = 2, /// < @todo: support npu
} BackendType;

/**
 * @brief Model type
 * @note  Enable only when your library supports the model
 */
typedef enum {
  CAUSAL_LM_MODEL_QWEN3_0_6B = 0,
} ModelType;

/**
 * @brief Configuration structure
 */
typedef struct {
  // Add configuration options here as needed
  bool use_chat_template; /// < @brief Whether to apply chat template to input
  bool debug_mode; /// < @brief Check model file validity during initialization
  bool verbose;    /// < @brief Whether to print output during generation
} Config;

/**
 * @brief Set global options
 * @param config Configuration object
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode setOptions(Config config);

/**
 * @brief Model Quantization type
 */
typedef enum {
  CAUSAL_LM_QUANTIZATION_UNKNOWN = 0,
  CAUSAL_LM_QUANTIZATION_W4A32 = 1,  ///< 4-bit weights, 32-bit activations
  CAUSAL_LM_QUANTIZATION_W16A16 = 2, ///< 16-bit weights, 16-bit activations
  CAUSAL_LM_QUANTIZATION_W8A16 = 3,  ///< 8-bit weights, 16-bit activations
  CAUSAL_LM_QUANTIZATION_W32A32 = 4, ///< 32-bit weights, 32-bit activations
} ModelQuantizationType;

/**
 * @brief Load a model
 * @param compute Backend compute type
 * @param modeltype Model type
 * @param quant_type Model quantization type
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode loadModel(BackendType compute, ModelType modeltype,
                               ModelQuantizationType quant_type);

/**
 * @brief Load a model from a custom directory path
 * @param compute Backend compute type
 * @param model_path Path to the model directory (containing config.json,
 * nntr_config.json, etc.)
 * @param quant_type Model quantization type
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode loadModelFromPath(BackendType compute,
                                       const char *model_path,
                                       ModelQuantizationType quant_type);

/**
 * @brief Get performance metrics of the last run
 * @param metrics Pointer to PerformanceMetrics struct to be filled
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics);

/**
 * @brief Run inference
 * @param inputTextPrompt Input prompt
 * @param outputText Buffer to store output text
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runModel(const char *inputTextPrompt,
                              const char **outputText);

/**
 * @brief Session handle for multi-session inference
 */
typedef int SessionHandle;

/**
 * @brief Create a new inference session
 *
 * The model must be loaded via loadModel() before creating a session.
 *
 * @param handle Pointer to store the new session handle
 * @param compute Backend compute type
 * @param modeltype Model type
 * @param quant_type Model quantization type
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode createSession(SessionHandle *handle, BackendType compute,
                                   ModelType modeltype,
                                   ModelQuantizationType quant_type);

/**
 * @brief Create a new inference session from a custom directory path
 *
 * The model must be loaded via loadModelFromPath() before creating a session.
 *
 * @param handle Pointer to store the new session handle
 * @param compute Backend compute type
 * @param model_path Path to the model directory
 * @param quant_type Model quantization type
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode createSessionFromPath(SessionHandle *handle,
                                           BackendType compute,
                                           const char *model_path,
                                           ModelQuantizationType quant_type);

/**
 * @brief Run inference on a session (synchronous)
 *
 * @param handle Session handle
 * @param inputTextPrompt Input prompt
 * @param outputText Buffer to store output text
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runSession(SessionHandle handle,
                                const char *inputTextPrompt,
                                const char **outputText);

/**
 * @brief Run inference on a session (asynchronous)
 *
 * @param handle Session handle
 * @param inputTextPrompt Input prompt
 * @param requestId Pointer to store the async request ID
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode runSessionAsync(SessionHandle handle,
                                     const char *inputTextPrompt,
                                     int *requestId);

/**
 * @brief Await the result of an async inference request
 *
 * @param requestId Request ID returned by runSessionAsync()
 * @param outputText Buffer to store output text
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode awaitResult(int requestId, const char **outputText);

/**
 * @brief Get performance metrics of a session
 *
 * @param handle Session handle
 * @param metrics Pointer to PerformanceMetrics struct to be filled
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode getSessionMetrics(SessionHandle handle,
                                       PerformanceMetrics *metrics);

/**
 * @brief Destroy a session and free its resources
 *
 * @param handle Session handle
 * @return ErrorCode
 */
WIN_EXPORT ErrorCode destroySession(SessionHandle handle);

#ifdef __cplusplus
}
#endif

#endif // __CAUSAL_LM_API_H__
