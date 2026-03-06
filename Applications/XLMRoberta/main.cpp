/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	main.cpp
 * @date	06 Jan 2026
 * @brief	This is a main file for XLM Roberta application
 * @see		https://github.com/nnstreamer/
 * @author	Junbong Yu <junbong.yu@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>

#include "causal_lm.h"

#include "xlm_roberta.h"
#include <sys/resource.h>

#include <atomic>
#include <chrono>
#include <thread>

using json = nlohmann::json;

std::atomic<size_t> peak_rss_kb{0};
std::atomic<bool> tracking_enabled{true};

void printMemoryUsage() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  std::cout << "Max Resident Set Size: " << usage.ru_maxrss << " KB"
            << std::endl;
}

size_t read_vm_rss_kb() {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.find("VmRSS:") == 0) {
      size_t kb = 0;
      sscanf(line.c_str(), "VmRSS: %zu kB", &kb);
      return kb;
    }
  }
  return 0;
}

size_t read_private_rss_kb() {
  std::ifstream smaps("/proc/self/smaps_rollup");
  std::string line;
  size_t total = 0;
  while (std::getline(smaps, line)) {
    if (line.find("Private_Clean:") == 0 || line.find("Private_Dirty:") == 0) {
      size_t kb;
      sscanf(line.c_str(), "%*s %zu", &kb);
      total += kb;
    }
  }
  return total;
}

void start_peak_tracker() {
  std::thread([] {
    while (tracking_enabled.load()) {
      size_t current = read_private_rss_kb();
      size_t prev = peak_rss_kb.load();
      if (current > prev) {
        peak_rss_kb.store(current);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }).detach();
}

void stop_and_print_peak() {
  tracking_enabled.store(false);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  std::cout << "Peak memory usage (VmRSS): " << peak_rss_kb.load() << " KB"
            << std::endl;
}

int main(int argc, char *argv[]) {

  /** Register all runnable causallm models to factory */
  causallm::Factory::Instance().registerModel(
    "LlamaForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::CausalLM>(cfg, generation_cfg,
                                                  nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "XLMRobertaModel", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<xlmroberta::XLMRoberta>(cfg, generation_cfg,
                                                      nntr_cfg);
    });

  // Validate arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path> [input_prompt]\n"
              << "  <model_path>   : Path to model directory\n"
              << "  [input_prompt] : Optional input text (uses sample_input if "
                 "omitted)\n";
    return EXIT_FAILURE;
  }

  const std::string model_path = argv[1];

  std::string system_head_prompt = "";
  std::string system_tail_prompt = "";

  try {
    // Load configuration files
    json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    json generation_cfg =
      causallm::LoadJsonFile(model_path + "/generation_config.json");
    json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

    std::vector<std::string> input_texts;
    if (argc >= 3) {
      input_texts.push_back(argv[2]);
    } else {
      std::vector<std::string> sample_input_keys;
      for (auto it = nntr_cfg.begin(); it != nntr_cfg.end(); ++it) {
        if (it.key().find("sample_input") == 0) {
          sample_input_keys.push_back(it.key());
        }
      }
      std::sort(sample_input_keys.begin(), sample_input_keys.end());

      for (const auto &key : sample_input_keys) {
        input_texts.push_back(nntr_cfg[key].get<std::string>());
      }
    }

    if (nntr_cfg.contains("system_prompt")) {
      system_head_prompt =
        nntr_cfg["system_prompt"]["head_prompt"].get<std::string>();
      system_tail_prompt =
        nntr_cfg["system_prompt"]["tail_prompt"].get<std::string>();
    }

    const std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();

    std::cout << weight_file << std::endl;

    auto model = causallm::Factory::Instance().create(
      cfg["architectures"].get<std::vector<std::string>>()[0], cfg,
      generation_cfg, nntr_cfg);

    model->initialize();
    model->load_weight(weight_file);

#ifdef PROFILE
    start_peak_tracker();
#endif
#if defined(_WIN32)
    std::vector<std::string> system_head_prompts(input_texts.size(), "");
    std::vector<std::string> system_tail_prompts(input_texts.size(), "");
    model->run(input_texts, false, system_head_prompts, system_tail_prompts);
#else
    std::vector<std::string> system_head_prompts(input_texts.size(), "");
    std::vector<std::string> system_tail_prompts(input_texts.size(), "");
    model->run(input_texts, false, system_head_prompts, system_tail_prompts);
#endif
#ifdef PROFILE
    stop_and_print_peak();
#endif
  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
