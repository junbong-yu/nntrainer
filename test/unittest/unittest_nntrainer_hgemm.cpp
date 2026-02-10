// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_hgemm.cpp
 * @date	10 February 2026
 * @brief	This is unittest for hgemm operations
 * @see		https://github.com/nntrainer/nntrainer
 * @author	OpenCode
 * @bug		No known bugs except for NYI items
 */

#include "nntrainer_test_util.h"
#include <cpu_backend.h>
#include <fallback_internal.h>
#include <fp16.h>
#include <gtest/gtest.h>
#include <hgemm.h>
#include <numeric>
#include <random>
#include <vector>

#include <chrono>
#include <iostream>
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;

template <typename T>
static inline double find_max_diff(T *src, T *src2, int M, int N) {
  float max_diff = 0;
  double err_sum = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      max_diff = std::max(max_diff, std::abs(static_cast<float>(
                                      src[i * N + j] - src2[i * N + j])));
      err_sum += std::abs(static_cast<float>(src[i * N + j] - src2[i * N + j]));
    }
  }
  return max_diff;
}

template <typename T = float>
float compute_mse(const uint32_t M, const uint32_t N, std::vector<T> &ref_dst,
                  std::vector<T> &dst, bool print = false) {
  auto mean_squared_error = mse<T, T>(ref_dst.data(), dst.data(), M * N);
  auto cos_sim = cosine_similarity<T, T>(ref_dst.data(), dst.data(), M * N);
  auto max_differ = find_max_diff<T>(ref_dst.data(), dst.data(), M, N);

  auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
  auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);
  if (print) {
    std::cout << "[INFO]            MSE: " << mean_squared_error
              << ", COS_SIM: " << cos_sim << ", MAX_DIFFER: " << max_differ
              << ", SUM: " << sum << ", SUM_GT: " << sum_gt << std::endl;
  }
  return mean_squared_error;
}

static std::vector<float> convert_fp16_to_fp32(const std::vector<__fp16> &fp16_vec) {
  std::vector<float> fp32_vec(fp16_vec.size());
  for (size_t i = 0; i < fp16_vec.size(); ++i) {
    fp32_vec[i] = static_cast<float>(fp16_vec[i]);
  }
  return fp32_vec;
}

static std::vector<__fp16> convert_fp32_to_fp16(const std::vector<float> &fp32_vec) {
  std::vector<__fp16> fp16_vec(fp32_vec.size());
  for (size_t i = 0; i < fp32_vec.size(); ++i) {
    fp16_vec[i] = static_cast<__fp16>(fp32_vec[i]);
  }
  return fp16_vec;
}

float test_hgemm(const uint32_t M, const uint32_t K, const uint32_t N,
                 const __fp16 *weights, const __fp16 *activations,
                 std::vector<__fp16> &ref_dst_fp16, bool TransA = false,
                 bool TransB = false, bool print = false) {
  std::vector<__fp16> dst_fp16(M * N);
  
  auto t1 = high_resolution_clock::now();
  hgemm(activations, weights, dst_fp16.data(), M, N, K, 1.0f, 0.0f, TransA,
        TransB);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] hgemm: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  auto mean_squared_error = compute_mse(M, N, ref_dst_fp16, dst_fp16, print);
  return mean_squared_error;
}

float test_hgemm_small(const uint32_t M, const uint32_t K, const uint32_t N,
                       const __fp16 *weights, const __fp16 *activations,
                       std::vector<__fp16> &ref_dst_fp16, bool TransA = false,
                       bool TransB = false, bool print = false) {
  std::vector<__fp16> dst_fp16(M * N);
  
  auto t1 = high_resolution_clock::now();
  hgemm_small(activations, weights, dst_fp16.data(), M, N, K, 1.0f, 0.0f, TransA,
              TransB);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] hgemm_small: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  auto mean_squared_error = compute_mse(M, N, ref_dst_fp16, dst_fp16, print);
  return mean_squared_error;
}

float test_hgemm_K1(const uint32_t M, const uint32_t K, const uint32_t N,
                    const __fp16 *weights, const __fp16 *activations,
                    std::vector<__fp16> &ref_dst_fp16, bool TransA = false,
                    bool TransB = false, bool print = false) {
  std::vector<__fp16> dst_fp16(M * N);
  
  auto t1 = high_resolution_clock::now();
  hgemm_K1(activations, weights, dst_fp16.data(), M, N, K, 1.0f, 0.0f, TransA,
           TransB);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] hgemm_K1: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  auto mean_squared_error = compute_mse(M, N, ref_dst_fp16, dst_fp16, print);
  return mean_squared_error;
}

static void run_hgemm_test(const uint32_t M, const uint32_t K, const uint32_t N,
                           float &hgemm_mse, float &hgemm_small_mse,
                           bool TransA = false, bool TransB = false,
                           bool print = false) {
  nntrainer::init_backend();

  if (print) {
    std::cout << "[INFO] HGEMM Test (M:" << M << ", K:" << K << ", N:" << N
              << ", TransA:" << TransA << ", TransB:" << TransB << ")"
              << std::endl;
  }

  std::vector<float> activation_f32 = generate_random_vector<float>(M * K);
  std::vector<float> weight_f32 = generate_random_vector<float>(N * K);
  
  std::vector<__fp16> activation_fp16 = convert_fp32_to_fp16(activation_f32);
  std::vector<__fp16> weight_fp16 = convert_fp32_to_fp16(weight_f32);
  
  std::vector<__fp16> ref_dst_fp16(M * N);
  std::vector<float> ref_dst_f32(M * N);

  auto t1 = high_resolution_clock::now();
  for (int tc = 0; tc < 10; ++tc) {
    nntrainer::sgemm(0, TransA, TransB, M, N, K, 1.F, activation_f32.data(), K,
                     weight_f32.data(), K, 0.F, ref_dst_f32.data(), N);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] sgemm (avg): " << dt.count() / 10 << " ns "
              << dt.count() / 10 / 1'000 << " us " << dt.count() / 10 / 1'000'000
              << " ms " << std::endl;
  }

  ref_dst_fp16 = convert_fp32_to_fp16(ref_dst_f32);

  hgemm_mse = test_hgemm(M, K, N, weight_fp16.data(), activation_fp16.data(),
                         ref_dst_fp16, TransA, TransB, print);
  hgemm_small_mse =
    test_hgemm_small(M, K, N, weight_fp16.data(), activation_fp16.data(),
                     ref_dst_fp16, TransA, TransB, print);
}

TEST(nntrainer_hgemm, basic_hgemm_256x256x256) {
  const unsigned int M = 256;
  const unsigned int K = 256;
  const unsigned int N = 256;
  float hgemm_mse, hgemm_small_mse;
  constexpr float eps = 1e-3;
  run_hgemm_test(M, K, N, hgemm_mse, hgemm_small_mse, false, false, true);
  ASSERT_LE(hgemm_mse, eps * M * K * N);
  ASSERT_LE(hgemm_small_mse, eps * M * K * N);
}

TEST(nntrainer_hgemm, basic_hgemm_512x512x512) {
  const unsigned int M = 512;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float hgemm_mse, hgemm_small_mse;
  constexpr float eps = 1e-3;
  run_hgemm_test(M, K, N, hgemm_mse, hgemm_small_mse, false, false, true);
  ASSERT_LE(hgemm_mse, eps * M * K * N);
  ASSERT_LE(hgemm_small_mse, eps * M * K * N);
}

TEST(nntrainer_hgemm, basic_hgemm_256x512x512) {
  const unsigned int M = 256;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float hgemm_mse, hgemm_small_mse;
  constexpr float eps = 1e-3;
  run_hgemm_test(M, K, N, hgemm_mse, hgemm_small_mse, false, false, true);
  ASSERT_LE(hgemm_mse, eps * M * K * N);
  ASSERT_LE(hgemm_small_mse, eps * M * K * N);
}

TEST(nntrainer_hgemm, basic_hgemm_512x256x128) {
  const unsigned int M = 512;
  const unsigned int K = 256;
  const unsigned int N = 128;
  float hgemm_mse, hgemm_small_mse;
  constexpr float eps = 1e-3;
  run_hgemm_test(M, K, N, hgemm_mse, hgemm_small_mse, false, false, true);
  ASSERT_LE(hgemm_mse, eps * M * K * N);
  ASSERT_LE(hgemm_small_mse, eps * M * K * N);
}

TEST(nntrainer_hgemm, basic_hgemm_128x256x512) {
  const unsigned int M = 128;
  const unsigned int K = 256;
  const unsigned int N = 512;
  float hgemm_mse, hgemm_small_mse;
  constexpr float eps = 1e-3;
  run_hgemm_test(M, K, N, hgemm_mse, hgemm_small_mse, false, false, true);
  ASSERT_LE(hgemm_mse, eps * M * K * N);
  ASSERT_LE(hgemm_small_mse, eps * M * K * N);
}

static void run_hgemm_transB_test(const uint32_t M, const uint32_t K,
                                   const uint32_t N, float &hgemm_mse,
                                   float &hgemm_small_mse, bool TransB = true,
                                   bool print = false) {
  nntrainer::init_backend();

  if (print) {
    std::cout << "[INFO] HGEMM TransB Test (M:" << M << ", K:" << K << ", N:" << N
              << ", TransB:" << TransB << ")" << std::endl;
  }

  std::vector<float> activation_f32 = generate_random_vector<float>(M * K);
  std::vector<float> weight_f32 = generate_random_vector<float>(N * K);
  
  std::vector<__fp16> activation_fp16 = convert_fp32_to_fp16(activation_f32);
  std::vector<__fp16> weight_fp16 = convert_fp32_to_fp16(weight_f32);
  
  std::vector<__fp16> ref_dst_fp16(M * N);
  std::vector<float> ref_dst_f32(M * N);

  auto t1 = high_resolution_clock::now();
  for (int tc = 0; tc < 10; ++tc) {
    nntrainer::sgemm(0, false, TransB, M, N, K, 1.F, activation_f32.data(), K,
                     weight_f32.data(), K, 0.F, ref_dst_f32.data(), N);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] sgemm (avg): " << dt.count() / 10 << " ns "
              << dt.count() / 10 / 1'000 << " us " << dt.count() / 10 / 1'000'000
              << " ms " << std::endl;
  }

  ref_dst_fp16 = convert_fp32_to_fp16(ref_dst_f32);

  hgemm_mse = test_hgemm(M, K, N, weight_fp16.data(), activation_fp16.data(),
                         ref_dst_fp16, false, TransB, print);
  hgemm_small_mse =
    test_hgemm_small(M, K, N, weight_fp16.data(), activation_fp16.data(),
                     ref_dst_fp16, false, TransB, print);
}

TEST(nntrainer_hgemm, hgemm_transB_256x256x256) {
  const unsigned int M = 256;
  const unsigned int K = 256;
  const unsigned int N = 256;
  float hgemm_mse, hgemm_small_mse;
  constexpr float eps = 1e-3;
  run_hgemm_transB_test(M, K, N, hgemm_mse, hgemm_small_mse, true, true);
  ASSERT_LE(hgemm_mse, eps * M * K * N);
  ASSERT_LE(hgemm_small_mse, eps * M * K * N);
}

TEST(nntrainer_hgemm, hgemm_transB_512x512x512) {
  const unsigned int M = 512;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float hgemm_mse, hgemm_small_mse;
  constexpr float eps = 1e-3;
  run_hgemm_transB_test(M, K, N, hgemm_mse, hgemm_small_mse, true, true);
  ASSERT_LE(hgemm_mse, eps * M * K * N);
  ASSERT_LE(hgemm_small_mse, eps * M * K * N);
}

static void run_hgemm_K1_test(const uint32_t M, const uint32_t K,
                              const uint32_t N, float &hgemm_K1_mse,
                              bool print = false) {
  nntrainer::init_backend();

  if (print) {
    std::cout << "[INFO] HGEMM K1 Test (M:" << M << ", K:" << K << ", N:" << N
              << ")" << std::endl;
  }

  std::vector<float> activation_f32 = generate_random_vector<float>(M * K);
  std::vector<float> weight_f32 = generate_random_vector<float>(N * K);
  
  std::vector<__fp16> activation_fp16 = convert_fp32_to_fp16(activation_f32);
  std::vector<__fp16> weight_fp16 = convert_fp32_to_fp16(weight_f32);
  
  std::vector<__fp16> ref_dst_fp16(M * N);
  std::vector<float> ref_dst_f32(M * N);

  auto t1 = high_resolution_clock::now();
  for (int tc = 0; tc < 10; ++tc) {
    nntrainer::sgemm(0, false, false, M, N, K, 1.F, activation_f32.data(), K,
                     weight_f32.data(), K, 0.F, ref_dst_f32.data(), N);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] sgemm (avg): " << dt.count() / 10 << " ns "
              << dt.count() / 10 / 1'000 << " us " << dt.count() / 10 / 1'000'000
              << " ms " << std::endl;
  }

  ref_dst_fp16 = convert_fp32_to_fp16(ref_dst_f32);

  hgemm_K1_mse = test_hgemm_K1(M, K, N, weight_fp16.data(), activation_fp16.data(),
                               ref_dst_fp16, false, false, print);
}

TEST(nntrainer_hgemm, hgemm_K1_256x1x256) {
  const unsigned int M = 256;
  const unsigned int K = 1;
  const unsigned int N = 256;
  float hgemm_K1_mse;
  constexpr float eps = 1e-3;
  run_hgemm_K1_test(M, K, N, hgemm_K1_mse, true);
  ASSERT_LE(hgemm_K1_mse, eps * M);
}

TEST(nntrainer_hgemm, hgemm_K1_512x1x512) {
  const unsigned int M = 512;
  const unsigned int K = 1;
  const unsigned int N = 512;
  float hgemm_K1_mse;
  constexpr float eps = 1e-3;
  run_hgemm_K1_test(M, K, N, hgemm_K1_mse, true);
  ASSERT_LE(hgemm_K1_mse, eps * M);
}

TEST(nntrainer_hgemm, hgemm_K1_128x1x512) {
  const unsigned int M = 128;
  const unsigned int K = 1;
  const unsigned int N = 512;
  float hgemm_K1_mse;
  constexpr float eps = 1e-3;
  run_hgemm_K1_test(M, K, N, hgemm_K1_mse, true);
  ASSERT_LE(hgemm_K1_mse, eps * M);
}
