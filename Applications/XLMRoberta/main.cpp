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

// std::atomic<size_t> peak_rss_kb{0};
// std::atomic<bool> tracking_enabled{true};

// void printMemoryUsage() {
//   struct rusage usage;
//   getrusage(RUSAGE_SELF, &usage);
//   std::cout << "Max Resident Set Size: " << usage.ru_maxrss << " KB"
//             << std::endl;
// }

// size_t read_vm_rss_kb() {
//   std::ifstream status("/proc/self/status");
//   std::string line;
//   while (std::getline(status, line)) {
//     if (line.find("VmRSS:") == 0) {
//       size_t kb = 0;
//       sscanf(line.c_str(), "VmRSS: %zu kB", &kb);
//       return kb;
//     }
//   }
//   return 0;
// }

// size_t read_private_rss_kb() {
//   std::ifstream smaps("/proc/self/smaps_rollup");
//   std::string line;
//   size_t total = 0;
//   while (std::getline(smaps, line)) {
//     if (line.find("Private_Clean:") == 0 || line.find("Private_Dirty:") == 0)
//     {
//       size_t kb;
//       sscanf(line.c_str(), "%*s %zu", &kb);
//       total += kb;
//     }
//   }
//   return total;
// }

// void start_peak_tracker() {
//   std::thread([] {
//     while (tracking_enabled.load()) {
//       size_t current = read_private_rss_kb();
//       size_t prev = peak_rss_kb.load();
//       if (current > prev) {
//         peak_rss_kb.store(current);
//       }
//       std::this_thread::sleep_for(std::chrono::milliseconds(10));
//     }
//   }).detach();
// }

// void stop_and_print_peak() {
//   tracking_enabled.store(false);
//   std::this_thread::sleep_for(std::chrono::milliseconds(20));
//   std::cout << "Peak memory usage (VmRSS): " << peak_rss_kb.load() << " KB"
//             << std::endl;
// }

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
  //   causallm::Factory::Instance().registerModel(
  //     "Qwen3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
  //       return std::make_unique<causallm::Qwen3CausalLM>(cfg, generation_cfg,
  //                                                        nntr_cfg);
  //     });
  //   causallm::Factory::Instance().registerModel(
  //     "Qwen3MoeForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg)
  //     {
  //       return std::make_unique<causallm::Qwen3MoECausalLM>(cfg,
  //       generation_cfg,
  //                                                           nntr_cfg);
  //     });
  //   causallm::Factory::Instance().registerModel(
  //     "Qwen3SlimMoeForCausalLM",
  //     [](json cfg, json generation_cfg, json nntr_cfg) {
  //       return std::make_unique<causallm::Qwen3SlimMoECausalLM>(
  //         cfg, generation_cfg, nntr_cfg);
  //     });
  //   causallm::Factory::Instance().registerModel(
  //     "Qwen3CachedSlimMoeForCausalLM",
  //     [](json cfg, json generation_cfg, json nntr_cfg) {
  //       return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
  //         cfg, generation_cfg, nntr_cfg);
  //     });
  //   causallm::Factory::Instance().registerModel(
  //     "GptOssForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
  //       return std::make_unique<causallm::GptOssForCausalLM>(cfg,
  //       generation_cfg,
  //                                                            nntr_cfg);
  //     });
  //   causallm::Factory::Instance().registerModel(
  //     "GptOssCachedSlimCausalLM",
  //     [](json cfg, json generation_cfg, json nntr_cfg) {
  //       return std::make_unique<causallm::GptOssCachedSlimCausalLM>(
  //         cfg, generation_cfg, nntr_cfg);
  //     });

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

    // // Determine input text
    // if (argc >= 3) {
    //   input_text = argv[2];
    // } else {
    //   input_text = nntr_cfg["sample_input"].get<std::string>();
    // }

    if (nntr_cfg.contains("system_prompt")) {
      system_head_prompt =
        nntr_cfg["system_prompt"]["head_prompt"].get<std::string>();
      system_tail_prompt =
        nntr_cfg["system_prompt"]["tail_prompt"].get<std::string>();
    }

    // Construct weight file path
    const std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();

    std::cout << weight_file << std::endl;

    //     // Initialize and run model
    auto model = causallm::Factory::Instance().create(
      cfg["architectures"].get<std::vector<std::string>>()[0], cfg,
      generation_cfg, nntr_cfg);

    model->initialize();
    std::cout << "(JBD) model init success" << std::endl;
    model->load_weight(weight_file);

#ifdef PROFILE
    start_peak_tracker();
#endif
#if defined(_WIN32)
    model->run(input_text.c_str(), generation_cfg["do_sample"],
               system_head_prompt.c_str(), system_tail_prompt.c_str());
#else

#define BATCH
#ifdef BATCH
    std::string input_text0 =
      "Instruct: Given a web search query, retrieve relevant passages that "
      "answer the query\nQuery: how much protein should a female eat";
    std::string input_text1 =
      "Instruct: Given a web search query, retrieve relevant passages that "
      "answer the query\nQuery: 南瓜的家常做法";
    std::string input_text2 =
      "As a general guideline, the CDC's average requirement of protein for "
      "women ages 19 to 70 is 46 grams per day. But, as you can see from this "
      "chart, you'll need to increase that if you're expecting or training for "
      "a marathon. Check out the chart below to see how much protein you "
      "should be eating each day";
    std::string input_text3 =
      "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: "
      "1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 "
      "2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 "
      "4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 "
      "原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 "
      "2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 "
      "4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 "
      "6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅";

    // std::string input_text0 = "My name is James";
    // std::string input_text1 = "hello";
    // std::string input_text2 = "good night";
    // std::string input_text3 = "I love you";    

    std::vector<std::string> input_texts;
    input_texts.push_back(input_text0);
    input_texts.push_back(input_text1);
    input_texts.push_back(input_text2);
    input_texts.push_back(input_text3);

    std::vector<std::string> system_head_prompts = {"", "", "", ""};
    std::vector<std::string> system_tail_prompts = {"", "", "", ""};

    model->run(input_texts, false, system_head_prompts, system_tail_prompts);
#else
    // token length of below example sentence is 229, this is identified in
    // config,json
    std::string input_text =
      "<s>1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: "
      "1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 "
      "2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 "
      "4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 "
      "原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 "
      "2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 "
      "4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 "
      "6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅</s>";

    std::string system_head_prompts = "";
    std::string system_tail_prompts = "";

    model->run(input_text, false, system_head_prompts, system_tail_prompts);
#endif // BATCH

#endif // _WIN32
#ifdef PROFILE
    stop_and_print_peak();
#endif
  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
