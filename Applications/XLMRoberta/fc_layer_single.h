// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   fc_layer_single.h
 * @date	1 Feb 2026
 * @brief	This is Custom FC Layer Class for XLMRoberta
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Junbong Yu <junbong.yu@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __FC_LAYER_SINGLE_H__
#define __FC_LAYER_SINGLE_H__

#include <common_properties.h>
#include <layer_impl.h>

namespace xlmroberta {

/**
 * @class   FullyConnectedSingleLayer
 * @brief   fully connected layer (processes only first sequence step in
 * incremental_forwarding)
 */
class FullyConnectedSingleLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  FullyConnectedSingleLayer();

  /**
   * @brief     Destructor of Fully Connected Layer
   */
  ~FullyConnectedSingleLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] FullyConnectedSingle &&
   */
  FullyConnectedSingleLayer(FullyConnectedSingleLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs FullyConnectedSingleLayer to be moved.
   */
  FullyConnectedSingleLayer &
  operator=(FullyConnectedSingleLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context,  unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   * @note
   * [note for LoRA] implicit calcDerivative is implicitly applied.
   * The weight is already updated with the LoRA's (W = W + W_lora)
   */
  void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return FullyConnectedSingleLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(nntrainer::RunLayerContext &context,
                unsigned int batch) override;

  static constexpr const char *type = "fully_connected_single";

private:
  float lora_scaling;
  std::tuple<nntrainer::props::Unit, nntrainer::props::LoraRank,
             nntrainer::props::LoraAlpha>
    fc_props;                             /**< fc layer properties :
                                                unit - number of output neurons,
                                                lora_rank - rank of lora (optional)
                                                lora_scaling - scaling factor of LoRA apply, i.e.,
                                             lora_scaling = alpha / lora_rank */
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
  std::array<unsigned int, 4> lora_idx;   /**< indices of the lora weights */
  std::unique_ptr<nntrainer::Quantizer> quantizer;
};
} // namespace xlmroberta

#endif /* __FC_LAYER_SINGLE_H__ */
