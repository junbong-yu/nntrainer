/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file	fc_layer_single.cpp
 * @date	1 Feb 2026
 * @brief	This is Custom FC Layer Class for XLMRoberta
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Junbong Yu <junbong.yu@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "fc_layer_single.h"
#include <common_properties.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <iostream>

namespace xlmroberta {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { weight, bias };
enum LORAParams { loraA, loraB, loraTmp, loraOut };

FullyConnectedSingleLayer::FullyConnectedSingleLayer() :
  nntrainer::LayerImpl(),
  lora_scaling(1.0f),
  fc_props(nntrainer::props::Unit(), nntrainer::props::LoraRank(),
           nntrainer::props::LoraAlpha()),
  quantizer(nullptr) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  lora_idx.fill(std::numeric_limits<unsigned>::max());
}

void FullyConnectedSingleLayer::finalize(nntrainer::InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<nntrainer::props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer =
    std::get<nntrainer::props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias =
    std::get<nntrainer::props::DisableBias>(*layer_impl_props);

  const auto &unit = std::get<nntrainer::props::Unit>(fc_props).get();
  const auto &lora_rank =
    (std::get<nntrainer::props::LoraRank>(fc_props).empty())
      ? 0
      : std::get<nntrainer::props::LoraRank>(fc_props).get();
  lora_scaling =
    (lora_rank && !std::get<nntrainer::props::LoraAlpha>(fc_props).empty())
      ? (float)std::get<nntrainer::props::LoraAlpha>(fc_props) / lora_rank
      : 1;

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  std::vector<nntrainer::TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  output_dims[0].height(1);
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  // @todo : This NCHW format setting is just temporal, it needs to be set by
  // global configuration

  /** Bias Dimension : (1, 1, 1, unit) */
  /// @note bias is directly added to activation
  /// since we have no dequantizer for add operation,
  /// we have to set its data type as same as activation.
  /// This should be updated when the dequantizer is supported.
  nntrainer::TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getActivationDataType()),
    is_nchw ? 0b0001 : 0b0100);

  /** Weight Dimension : (1, 1, in_dim.width(), unit)*/
  nntrainer::TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[FCParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[FCParams::bias] = context.requestWeight(
      bias_dim, bias_initializer, nntrainer::WeightRegularizer::NONE, 1.0f,
      bias_decay, "bias", true);
  }

  /** create weights for LoRA */
  if (lora_rank) {

    /** loraA Dimension : (1, 1, in_dim.width, lora_rank) */
    nntrainer::TensorDim loraA_dim(
      1, is_nchw ? 1 : lora_rank, is_nchw ? in_dim.width() : 1,
      is_nchw ? lora_rank : in_dim.channel(),
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()),
      is_nchw ? 0b0011 : 0b0101);

    /** loraB Dimension : (1, 1, lora_rank, unit) */
    nntrainer::TensorDim loraB_dim(
      1, is_nchw ? 1 : unit, is_nchw ? lora_rank : 1,
      is_nchw ? unit : lora_rank,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()),
      is_nchw ? 0b0011 : 0b0101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), lora_rank) */
    nntrainer::TensorDim loraTmp_dim(
      in_dim.batch(), is_nchw ? 1 : lora_rank, is_nchw ? in_dim.height() : 1,
      is_nchw ? lora_rank : in_dim.width(),
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getActivationDataType()),
      is_nchw ? 0b1011 : 0b1101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), unit) */
    nntrainer::TensorDim loraOut_dim(
      in_dim.batch(), is_nchw ? 1 : unit, is_nchw ? in_dim.height() : 1,
      is_nchw ? unit : in_dim.width(),
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getActivationDataType()),
      is_nchw ? 0b1011 : 0b1101);

    lora_idx[LORAParams::loraA] = context.requestWeight(
      loraA_dim, nntrainer::Initializer::ZEROS, weight_regularizer,
      weight_regularizer_constant, weight_decay, "loraA", true);

    lora_idx[LORAParams::loraB] = context.requestWeight(
      loraB_dim, nntrainer::Initializer::LECUN_NORMAL, weight_regularizer,
      weight_regularizer_constant, weight_decay, "loraB", true);

    lora_idx[LORAParams::loraTmp] = context.requestTensor(
      loraTmp_dim, "hidden_tmp_lora", nntrainer::Initializer::NONE, true,
      nntrainer::TensorLifespan::FORWARD_GRAD_LIFESPAN);

    lora_idx[LORAParams::loraOut] = context.requestTensor(
      loraOut_dim, "hidden_lora", nntrainer::Initializer::NONE, true,
      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  }

  ///@todo this quantizaer should be moved to tensor, not layer!
  switch (context.getWeightDataType()) {
  case ml::train::TensorDim::DataType::QINT4:
  case ml::train::TensorDim::DataType::QINT8:
  case ml::train::TensorDim::DataType::QINT16:
    quantizer = nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);
    break;
  default:
    quantizer = nullptr;
    break;
  }
}

void FullyConnectedSingleLayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void FullyConnectedSingleLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void FullyConnectedSingleLayer::setBatch(nntrainer::RunLayerContext &context,
                                         unsigned int batch) {
  if (!std::get<nntrainer::props::LoraRank>(fc_props).empty()) {
    // update Lora Tensor's batch info.
    context.updateTensor(lora_idx[LORAParams::loraTmp], batch);
    context.updateTensor(lora_idx[LORAParams::loraOut], batch);
  }
}

void FullyConnectedSingleLayer::forwarding(nntrainer::RunLayerContext &context,
                                           bool training) {
  nntrainer::Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  ///@todo This dequantization action should be moved to tensor.dot()
  if (quantizer != nullptr) {
    nntrainer::Tensor weight_ =
      quantizer->dequantize(weight, input_.getDataType());
    input_.dot(weight_, hidden_, false, false);
  } else {
    input_.dot(weight, hidden_, false, false);
  }

  if (!std::get<nntrainer::props::LoraRank>(fc_props).empty()) {
    nntrainer::Tensor &loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    nntrainer::Tensor &loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    nntrainer::Tensor &hidden_tmp_lora =
      context.getTensor(lora_idx[LORAParams::loraTmp]);
    nntrainer::Tensor &hidden_out_lora =
      context.getTensor(lora_idx[LORAParams::loraOut]);

    input_.dot(loraA, hidden_tmp_lora, false, false);
    hidden_tmp_lora.dot(loraB, hidden_out_lora, false, false);
    hidden_out_lora.multiply_i(lora_scaling);
    hidden_.add_i(hidden_out_lora);
  }

  if (auto &disable_bias =
        std::get<nntrainer::props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    nntrainer::Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_.add_i(bias);
  }
}

void FullyConnectedSingleLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor loraA, loraB, hidden_tmp_lora, hidden_out_lora;

  if (!std::get<nntrainer::props::LoraRank>(fc_props).empty()) {
    loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    hidden_tmp_lora = context.getTensor(lora_idx[LORAParams::loraTmp]);
    hidden_out_lora = context.getTensor(lora_idx[LORAParams::loraOut]);
  }

  nntrainer::TensorDim input_dim = input_.getDim();
  nntrainer::TensorDim hidden_dim = hidden_.getDim();

  nntrainer::TensorDim input_step_dim = input_dim;
  nntrainer::TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  input_step_dim.height(1);
  hidden_step_dim.batch(1);
  hidden_step_dim.height(1);

  // @todo make it parallelized with batch axis
  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    nntrainer::Tensor input_step = input_.getSharedDataTensor(
      input_step_dim, b * input_dim.getFeatureLen() + from * input_dim.width(),
      true);
    nntrainer::Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim,
      b * hidden_dim.getFeatureLen() + from * hidden_dim.width(), true);

    input_step.dot(weight, hidden_step, false, false);

    if (!std::get<nntrainer::props::LoraRank>(fc_props).empty()) {
      nntrainer::TensorDim hidden_tmp_lora_step_dim = hidden_tmp_lora.getDim();
      hidden_tmp_lora_step_dim.batch(1);
      hidden_tmp_lora_step_dim.height(1);
      nntrainer::TensorDim hidden_out_lora_step_dim = hidden_out_lora.getDim();
      hidden_out_lora_step_dim.batch(1);
      hidden_out_lora_step_dim.height(1);

      nntrainer::Tensor hidden_tmp_lora_step =
        hidden_tmp_lora.getSharedDataTensor(hidden_tmp_lora_step_dim,
                                            b * hidden_tmp_lora.height() *
                                                hidden_tmp_lora.width() +
                                              from * hidden_tmp_lora.width(),
                                            true);
      nntrainer::Tensor hidden_out_lora_step =
        hidden_out_lora.getSharedDataTensor(hidden_out_lora_step_dim,
                                            b * hidden_out_lora.height() *
                                                hidden_out_lora.width() +
                                              from * hidden_out_lora.width(),
                                            true);

      input_step.dot(loraA, hidden_tmp_lora_step, false, false);
      hidden_tmp_lora_step.dot(loraB, hidden_out_lora_step, false, false);
      hidden_out_lora_step.multiply_i(lora_scaling);
      hidden_step.add_i(hidden_out_lora_step);
    }

    if (auto &disable_bias =
          std::get<nntrainer::props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      nntrainer::Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
      hidden_step.add_i(bias);
    }
  }
}

void FullyConnectedSingleLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  nntrainer::Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);

  const nntrainer::Tensor &derivative_ =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  if (!std::get<nntrainer::props::LoraRank>(fc_props).empty()) {
    nntrainer::Tensor &lora_A = context.getWeight(lora_idx[LORAParams::loraA]);
    nntrainer::Tensor &lora_B = context.getWeight(lora_idx[LORAParams::loraB]);
    ret_.dot_deriv_wrt_1(weight.add(lora_A.dot(lora_B).multiply(lora_scaling)),
                         derivative_, false, false);
  } else {
    ret_.dot_deriv_wrt_1(weight, derivative_, false, false);
  }
}

void FullyConnectedSingleLayer::calcGradient(
  nntrainer::RunLayerContext &context) {

  /** (default) calcGradient - compute gradient of weight and bias */
  if (std::get<nntrainer::props::LoraRank>(fc_props).empty()) {
    nntrainer::Tensor &djdw =
      context.getWeightGrad(weight_idx[FCParams::weight]);
    djdw.setZero();

    const nntrainer::Tensor &derivative_ =
      context.getIncomingDerivative(SINGLE_INOUT_IDX);
    nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

    if (auto &disable_bias =
          std::get<nntrainer::props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      nntrainer::Tensor &djdb =
        context.getWeightGrad(weight_idx[FCParams::bias]);
      djdb.setZero();

      if (context.isGradientFirstAccess(weight_idx[FCParams::bias])) {
        derivative_.sum({0, 1, 2}, djdb);
      } else {
        /// @todo optimize below by adding beta to Tensor::sum
        nntrainer::Tensor t = derivative_.sum({0, 1, 2});
        djdb.add_i(t);
      }
    }

    input_.dot_deriv_wrt_2(
      djdw, derivative_, false, false,
      !context.isGradientFirstAccess(weight_idx[FCParams::weight]));
  } else {
    /** (lora) calcGradient - compute gradients of LoRA params only */
    nntrainer::Tensor &djdla =
      context.getWeightGrad(lora_idx[LORAParams::loraA]);
    nntrainer::Tensor &djdlb =
      context.getWeightGrad(lora_idx[LORAParams::loraB]);
    nntrainer::Tensor &djdtmp =
      context.getTensorGrad(lora_idx[LORAParams::loraTmp]);

    const nntrainer::Tensor &derivative_ =
      context.getIncomingDerivative(SINGLE_INOUT_IDX);
    nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
    nntrainer::Tensor &loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    nntrainer::Tensor &loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    nntrainer::Tensor &loraTmp =
      context.getTensor(lora_idx[LORAParams::loraTmp]);
    const auto &lora_derivative_ = derivative_.multiply(lora_scaling);

    loraTmp.dot_deriv_wrt_2(
      djdlb, lora_derivative_, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraB]));
    djdtmp.dot_deriv_wrt_1(
      loraB, lora_derivative_, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraTmp]));
    input_.dot_deriv_wrt_2(
      djdla, djdtmp, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraA]));
  }
}

} /* namespace xlmroberta */
