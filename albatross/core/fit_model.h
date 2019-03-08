/*
 * Copyright (C) 2019 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_CORE_FIT_MODEL_H
#define ALBATROSS_CORE_FIT_MODEL_H

namespace albatross {

template <typename ModelType, typename Fit> class FitModel {
public:
  template <typename X, typename Y, typename Z> friend class Prediction;

  static_assert(
      std::is_move_constructible<Fit>::value,
      "Fit type must be move constructible to avoid unexpected copying.");

  FitModel(const ModelType &model, Fit &fit) = delete;

  FitModel(const ModelType &model, Fit &&fit)
      : model_(model), fit_(std::move(fit)) {}

  template <typename PredictFeatureType>
  Prediction<ModelType, PredictFeatureType, Fit>
  get_prediction(const std::vector<PredictFeatureType> &features) const {
    return Prediction<ModelType, PredictFeatureType, Fit>(model_, fit_,
                                                          features);
  }

private:
  const ModelType model_;
  const Fit fit_;
};
}
#endif
