// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   test_api.cpp
 * @date   21 Jan 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @brief  Simple application to test CausalLM API with multi-thread support
 *
 */

#include "causal_lm_api.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr const char *COLOR_RESET = "\033[0m";
constexpr const char *COLOR_BOLD = "\033[1m";
constexpr const char *COLOR_CYAN = "\033[36m";
constexpr const char *COLOR_GREEN = "\033[32m";
constexpr const char *COLOR_YELLOW = "\033[33m";
constexpr const char *COLOR_BLUE = "\033[34m";
constexpr const char *COLOR_RED = "\033[31m";
constexpr const char *COLOR_MAGENTA = "\033[35m";

void printLine(const std::string &s, int length = 80) {
  for (int i = 0; i < length; ++i)
    std::cout << s;
  std::cout << std::endl;
}

void printSection(const std::string &section) {
  std::cout << "\n"
            << COLOR_BOLD << COLOR_BLUE
            << "+-------------------------------------------------------------+"
            << COLOR_RESET << "\n";
  std::cout << COLOR_BOLD << COLOR_BLUE << "|  " << section
            << std::string(58 - section.length(), ' ') << "|" << COLOR_RESET
            << "\n";
  std::cout << COLOR_BOLD << COLOR_BLUE
            << "+-------------------------------------------------------------+"
            << COLOR_RESET << "\n\n";
}

void printSuccess(const std::string &msg) {
  std::cout << COLOR_GREEN << "✓ " << COLOR_BOLD << msg << COLOR_RESET
            << "\n\n";
}

void printError(const std::string &msg) {
  std::cerr << COLOR_RED << "✗ " << COLOR_BOLD << "Error: " << COLOR_RESET
            << msg << "\n";
}

void printInfo(const std::string &label, const std::string &value) {
  std::cout << COLOR_CYAN << "  " << label << ":" << COLOR_RESET << " " << value
            << "\n";
}

void printLogo() {
  std::cout << "\n";
  std::cout << COLOR_BOLD << COLOR_MAGENTA;
  std::cout << "  ███╗   ██╗███╗   ██╗████████╗██████╗ \n";
  std::cout << "  ████╗  ██║████╗  ██║╚══██╔══╝██╔══██╗\n";
  std::cout << "  ██╔██╗ ██║██╔██╗ ██║   ██║   ██████╔╝\n";
  std::cout << "  ██║╚██╗██║██║╚██╗██║   ██║   ██╔══██╗\n";
  std::cout << "  ██║ ╚████║██║ ╚████║   ██║   ██║  ██║\n";
  std::cout << "  ╚═╝  ╚═══╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝\n";
  std::cout << COLOR_RESET;
  std::cout << COLOR_BOLD << COLOR_CYAN
            << "  ────────────────────────────────\n";
  std::cout << "      Causal Language Model API\n"
            << "  ────────────────────────────────\n";
  std::cout << COLOR_RESET << "\n";
}

void printUsage(const char *program_name) {
  std::cout << COLOR_YELLOW << "Usage:" << COLOR_RESET << "\n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " <model_path> [num_threads] [use_chat_template] [quantization] "
               "[verbose]\n\n";

  std::cout << COLOR_CYAN << "Arguments:" << COLOR_RESET << "\n";
  std::cout << "  model_path        " << COLOR_BOLD << "REQUIRED" << COLOR_RESET
            << "  - Path to model directory (e.g., /path/to/qwen3-0.6b)\n";
  std::cout << "  num_threads       " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET
            << "  - Number of concurrent sessions (default: 4)\n";
  std::cout << "  use_chat_template " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET << "  - 0/1 or true/false (default: 1)\n";
  std::cout << "  quantization      " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET
            << "  - W4A32/W16A16/W8A16/W32A32/UNKNOWN (default: UNKNOWN)\n";
  std::cout << "  verbose           " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET << "  - 0/1 or true/false (default: 0)\n\n";

  std::cout << COLOR_YELLOW << "Examples:" << COLOR_RESET << "\n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " /path/to/qwen3-0.6b 4 1 W32A32\n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " /path/to/qwen3-0.6b 8 0 UNKNOWN 1\n\n";
}

struct ThreadResult {
  int thread_id;
  std::string prompt;
  std::string output;
  ErrorCode error_code;
  PerformanceMetrics metrics;
  double elapsed_ms;
};

} // namespace

int main(int argc, char *argv[]) {
  printLogo();

  if (argc < 2) {
    printSection("ERROR: Missing Required Arguments");
    printUsage(argv[0]);
    return 1;
  }

  const char *model_path = argv[1];
  int num_threads = (argc >= 3) ? std::stoi(argv[2]) : 4;
  if (num_threads <= 0)
    num_threads = 4;

  bool use_chat_template = true;
  if (argc >= 4) {
    use_chat_template =
      (std::string(argv[3]) == "1" || std::string(argv[3]) == "true");
  }

  std::string quant_str = "UNKNOWN";
  ModelQuantizationType quant_type = CAUSAL_LM_QUANTIZATION_UNKNOWN;
  if (argc >= 5) {
    quant_str = std::string(argv[4]);
    if (quant_str == "W4A32")
      quant_type = CAUSAL_LM_QUANTIZATION_W4A32;
    else if (quant_str == "W16A16")
      quant_type = CAUSAL_LM_QUANTIZATION_W16A16;
    else if (quant_str == "W8A16")
      quant_type = CAUSAL_LM_QUANTIZATION_W8A16;
    else if (quant_str == "W32A32")
      quant_type = CAUSAL_LM_QUANTIZATION_W32A32;
  }

  bool verbose = false;
  if (argc >= 6) {
    verbose = (std::string(argv[5]) == "1" || std::string(argv[5]) == "true");
  }

  printSection("Configuration");
  printInfo("Model Path", model_path);
  printInfo("Num Threads", std::to_string(num_threads));
  printInfo("Use Chat Template", use_chat_template ? "true" : "false");
  printInfo("Quantization", quant_str);
  printInfo("Verbose", verbose ? "true" : "false");
  std::cout << "\n";

  printSection("Initialization");
  std::cout << COLOR_CYAN << "⏳ " << COLOR_RESET << "Configuring options...\n";
  Config config;
  config.use_chat_template = use_chat_template;
  config.debug_mode = true;
  config.verbose = verbose;
  ErrorCode err = setOptions(config);
  if (err != CAUSAL_LM_ERROR_NONE) {
    printError("Failed to set options");
    std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
    return 1;
  }
  printSuccess("Options configured successfully");

  printSection("Model Loading");
  std::cout << COLOR_CYAN << "⏳ " << COLOR_RESET
            << "Loading model from: " << COLOR_BOLD << model_path << COLOR_RESET
            << "\n";

  err = loadModelFromPath(CAUSAL_LM_BACKEND_CPU, model_path, quant_type);

  if (err != CAUSAL_LM_ERROR_NONE) {
    printError("Failed to load model");
    std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
    return 1;
  }
  printSuccess("Model loaded successfully");

  // Different prompts for each thread
  std::vector<std::string> prompts = {
    "Hello! How are you today?",
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about technology.",
    "What are the benefits of exercise?",
    "Describe the taste of chocolate.",
    "How do airplanes fly?",
    "What is machine learning?",
    "Tell me an interesting fact about space.",
    "Why is the sky blue?",
  };

  if (num_threads > (int)prompts.size()) {
    size_t original_size = prompts.size();
    for (int i = original_size; i < num_threads; ++i) {
      prompts.push_back(prompts[i % original_size]);
    }
  }

  printSection("Multi-Threaded Inference Test");
  std::cout << COLOR_CYAN << "⏳ " << COLOR_RESET << "Creating " << num_threads
            << " sessions and running concurrently...\n\n";

  std::vector<ThreadResult> results(num_threads);
  std::vector<std::thread> threads;
  std::atomic<int> completed_count{0};

  auto start_all = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      SessionHandle session = 0;
      ErrorCode err = createSessionFromPath(&session, CAUSAL_LM_BACKEND_CPU,
                                            model_path, quant_type);
      if (err != CAUSAL_LM_ERROR_NONE) {
        results[i].thread_id = i;
        results[i].error_code = err;
        results[i].prompt = prompts[i];
        std::cerr << COLOR_RED << "✗ Thread " << i
                  << ": Failed to create session (error="
                  << static_cast<int>(err) << ")" << COLOR_RESET << "\n";
        return;
      }

      const std::string &prompt = prompts[i];
      results[i].thread_id = i;
      results[i].prompt = prompt;
      results[i].error_code = CAUSAL_LM_ERROR_NONE;

      std::string out_filename =
        std::string("session") + std::to_string(i) + ".txt";
      std::ofstream out_file(out_filename);

      auto start = std::chrono::high_resolution_clock::now();

      const char *output = nullptr;
      err = runSession(session, prompt.c_str(), &output);

      auto end = std::chrono::high_resolution_clock::now();
      results[i].elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

      if (err == CAUSAL_LM_ERROR_NONE && output) {
        results[i].output = std::string(output);
      } else {
        results[i].error_code = err;
      }

      PerformanceMetrics metrics;
      err = getSessionMetrics(session, &metrics);
      if (err == CAUSAL_LM_ERROR_NONE) {
        results[i].metrics = metrics;
      }

      if (out_file.is_open()) {
        out_file << "Session: " << i << "\n";
        out_file << "Prompt: " << prompt << "\n\n";
        if (results[i].error_code == CAUSAL_LM_ERROR_NONE) {
          out_file << "Output:\n" << results[i].output << "\n\n";
          out_file << "Performance Metrics:\n";
          out_file << "  Prefill tokens: " << metrics.prefill_tokens << "\n";
          out_file << "  Prefill duration: " << std::fixed
                   << std::setprecision(2) << metrics.prefill_duration_ms
                   << " ms\n";
          out_file << "  Generation tokens: " << metrics.generation_tokens
                   << "\n";
          out_file << "  Generation duration: " << std::fixed
                   << std::setprecision(2) << metrics.generation_duration_ms
                   << " ms\n";
          out_file << "  Total duration: " << std::fixed << std::setprecision(2)
                   << metrics.total_duration_ms << " ms\n";
          out_file << "  Peak memory: " << metrics.peak_memory_kb / 1024
                   << " MB\n";
          out_file << "  Thread elapsed: " << std::fixed << std::setprecision(2)
                   << results[i].elapsed_ms << " ms\n";
        } else {
          out_file << "Error: Inference failed (code="
                   << static_cast<int>(results[i].error_code) << ")\n";
        }
        out_file.close();
      }

      int completed = ++completed_count;
      std::cout << COLOR_GREEN << "✓ Thread " << i << " completed ("
                << completed << "/" << num_threads << ")" << COLOR_RESET
                << " -> " << out_filename << "\n";

      destroySession(session);
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  auto end_all = std::chrono::high_resolution_clock::now();
  double total_elapsed_ms =
    std::chrono::duration<double, std::milli>(end_all - start_all).count();

  printSection("Results Summary");
  std::cout << COLOR_CYAN << "  Total elapsed time: " << COLOR_RESET
            << std::fixed << std::setprecision(2) << total_elapsed_ms
            << " ms\n";
  std::cout << COLOR_CYAN << "  Number of sessions: " << COLOR_RESET
            << num_threads << "\n";
  std::cout << COLOR_CYAN << "  Average per session: " << COLOR_RESET
            << std::fixed << std::setprecision(2)
            << total_elapsed_ms / num_threads << " ms\n\n";

  for (int i = 0; i < num_threads; ++i) {
    const auto &r = results[i];
    std::cout << "  " << COLOR_BOLD << "Session " << i << ":" << COLOR_RESET
              << "\n";
    std::cout << "    Prompt: " << COLOR_YELLOW << r.prompt << COLOR_RESET
              << "\n";
    if (r.error_code == CAUSAL_LM_ERROR_NONE) {
      double prefill_tps =
        r.metrics.prefill_duration_ms > 0
          ? (r.metrics.prefill_tokens / r.metrics.prefill_duration_ms * 1000.0)
          : 0.0;
      double gen_tps = r.metrics.generation_duration_ms > 0
                         ? (r.metrics.generation_tokens /
                            r.metrics.generation_duration_ms * 1000.0)
                         : 0.0;

      std::cout << "    Tokens: " << r.metrics.generation_tokens
                << " (prefill: " << r.metrics.prefill_tokens << ")\n";
      std::cout << "    Throughput: " << std::fixed << std::setprecision(1)
                << prefill_tps << " / " << gen_tps << " tokens/sec\n";
      std::cout << "    Duration: " << std::fixed << std::setprecision(2)
                << r.elapsed_ms << " ms\n";
      std::cout << "    Output file: session" << i << ".txt\n";
    } else {
      std::cout << "    " << COLOR_RED
                << "FAILED (error=" << static_cast<int>(r.error_code) << ")"
                << COLOR_RESET << "\n";
    }
    std::cout << "\n";
  }

  printLine("═", 63);
  std::cout << COLOR_BOLD << COLOR_GREEN << "  ✓ Multi-threaded test completed!"
            << COLOR_RESET << "\n";
  printLine("═", 63);
  std::cout << "\n";

  return 0;
}
