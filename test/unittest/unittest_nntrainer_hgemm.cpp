// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_hgemm.cpp
 * @date	10 February 2026
 * @brief	This is unittest for hgemm operations with pure __fp16
 * @see		https://github.com/nntrainer/nntrainer
 * @author	OpenCode
 * @bug		No known bugs except for NYI items
 */

#include "nntrainer_test_util.h"
#include <cpu_backend.h>
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
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      max_diff =
        std::max(max_diff, std::abs(static_cast<float>(src[i * N + j] -
                                                        src2[i * N + j])));
    }
  }
  return max_diff;
}

template <typename T = __fp16>
float compute_mse(const uint32_t M, const uint32_t N, std::vector<T> &ref_dst,
                  std::vector<T> &dst, bool print = false) {
  auto mean_squared_error = mse<T, T>(ref_dst.data(), dst.data(), M * N);
  auto cos_sim = cosine_similarity<T, T>(ref_dst.data(), dst.data(), M * N);
  auto max_differ = find_max_diff<T>(ref_dst.data(), dst.data(), M, N);

  float sum = 0;
  float sum_gt = 0;
  for (size_t i = 0; i < dst.size(); ++i) {
    sum += static_cast<float>(dst[i]);
    sum_gt += static_cast<float>(ref_dst[i]);
  }

  if (print) {
    std::cout << "[INFO]            MSE: " << mean_squared_error
              << ", COS_SIM: " << cos_sim << ", MAX_DIFFER: " << max_differ
              << ", SUM: " << sum << ", SUM_GT: " << sum_gt << std::endl;
  }
  return mean_squared_error;
}

static std::vector<__fp16> generate_random_fp16(size_t size) {
  std::vector<float> f32_vec = generate_random_vector<float>(size);
  std::vector<__fp16> fp16_vec(size);
  for (size_t i = 0; i < size; ++i) {
    fp16_vec[i] = static_cast<__fp16>(f32_vec[i]);
  }
  return fp16_vec;
}

float test_hgemm(const uint32_t M, const uint32_t K, const uint32_t N,
                  const __fp16 *weights, const __fp16 *activations,
                  std::vector<__fp16> &ref_dst, bool TransA = false,
                  bool TransB = false, bool print = false) {
  std::vector<__fp16> dst(M * N);

  auto t1 = high_resolution_clock::now();
  hgemm(activations, weights, dst.data(), M, N, K, 1.0f, 0.0f, TransA, TransB);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] hgemm: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);
  return mean_squared_error;
}

float test_hgemm_small(const uint32_t M, const uint32_t K, const uint32_t N,
                        const __fp16 *weights, const __fp16 *activations,
                        std::vector<__fp16> &ref_dst, bool TransA = false,
                        bool TransB = false, bool print = false) {
  std::vector<__fp16> dst(M * N);

  auto t1 = high_resolution_clock::now();
  hgemm_small(activations, weights, dst.data(), M, N, K, 1.0f, 0.0f, TransA,
              TransB);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] hgemm_small: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);
  return mean_squared_error;
}

float test_hgemm_K1(const uint32_t M, const uint32_t K, const uint32_t N,
                     const __fp16 *weights, const __fp16 *activations,
                     std::vector<__fp16> &ref_dst, bool TransA = false,
                     bool TransB = false, bool print = false) {
  std::vector<__fp16> dst(M * N);

  auto t1 = high_resolution_clock::now();
  hgemm_K1(activations, weights, dst.data(), M, N, K, 1.0f, 0.0f, TransA, TransB);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] hgemm_K1: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);
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

  std::vector<__fp16> activation_fp16 = generate_random_fp16(M * K);
  std::vector<__fp16> weight_fp16 = generate_random_fp16(N * K);
  std::vector<__fp16> ref_dst(M * N);

  auto t1 = high_resolution_clock::now();
  for (int tc = 0; tc < 10; ++tc) {
    hgemm(activation_fp16.data(), weight_fp16.data(), ref_dst.data(), M, N, K,
           1.0f, 0.0f, TransA, TransB);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] hgemm ref (avg): " << dt.count() / 10 << " ns "
              << dt.count() / 10 / 1'000 << " us "
              << dt.count() / 10 / 1'000'000 << " ms " << std::endl;
  }

  hgemm_mse =
    test_hgemm(M, K, N, weight_fp16.data(), activation_fp16.data(), ref_dst,
               TransA, TransB, print);
  hgemm_small_mse =
    test_hgemm_small(M, K, N, weight_fp16.data(), activation_fp16.data(),
                     ref_dst, TransA, TransB, print);
}

TEST(nntrainer_hgemm, basic_hgemm_1024x3072x3072) {
  const unsigned int M = 1024;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float hgemm_mse, hgemm_small_mse;
  constexpr float eps = 1e-5;
  run_hgemm_test(M, K, N, hgemm_mse, hgemm_small_mse, false, false, true);
  ASSERT_LE(hgemm_mse, eps);
  ASSERT_LE(hgemm_small_mse, eps);
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

  std::vector<__fp16> activation_fp16 = generate_random_fp16(M * K);
  std::vector<__fp16> weight_fp16 = generate_random_fp16(N * K);
  std::vector<__fp16> ref_dst(M * N);

  auto t1 = high_resolution_clock::now();
  for (int tc = 0; tc < 10; ++tc) {
    hgemm(activation_fp16.data(), weight_fp16.data(), ref_dst.data(), M, N, K,
           1.0f, 0.0f, false, TransB);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] hgemm ref (avg): " << dt.count() / 10 << " ns "
              << dt.count() / 10 / 1'000 << " us "
              << dt.count() / 10 / 1'000'000 << " ms " << std::endl;
  }

  hgemm_mse =
    test_hgemm(M, K, N, weight_fp16.data(), activation_fp16.data(), ref_dst,
               false, TransB, print);
  hgemm_small_mse =
    test_hgemm_small(M, K, N, weight_fp16.data(), activation_fp16.data(),
                     ref_dst, false, TransB, print);
}

TEST(nntrainer_hgemm, hgemm_transB_1024x3072x3072) {
  const unsigned int M = 1024;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float hgemm_mse, hgemm_small_mse;
  constexpr float eps = 1e-5;
  run_hgemm_transB_test(M, K, N, hgemm_mse, hgemm_small_mse, true, true);
  ASSERT_LE(hgemm_mse, eps);
  ASSERT_LE(hgemm_small_mse, eps);
}

GTEST_API_ int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}

