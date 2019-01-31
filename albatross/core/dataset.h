/*
 * Copyright (C) 2018 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_CORE_DATASET_H
#define ALBATROSS_CORE_DATASET_H

#include "core/distribution.h"
#include "core/traits.h"
#include <Eigen/Core>
#include <cereal/archives/json.hpp>
#include <map>
#include <vector>

namespace albatross {

// A JointDistribution has a dense covariance matrix, which
// contains the covariance between each variable and all others.
using JointDistribution = Distribution<Eigen::MatrixXd>;

// We use a wrapper around DiagonalMatrix in order to make
// the resulting distribution serializable
using DiagonalMatrixXd =
    Eigen::SerializableDiagonalMatrix<double, Eigen::Dynamic>;
// A MarginalDistribution has only a digaonal covariance
// matrix, so in turn only describes the variance of each
// variable independent of all others.
using MarginalDistribution = Distribution<DiagonalMatrixXd>;

/*
 * A RegressionDataset holds two vectors of data, the features
 * where a single feature can be any class that contains the information used
 * to make predictions of the target.  This is called a RegressionDataset since
 * it is assumed that each feature is regressed to a single double typed
 * target.
 */
template <typename FeatureType> struct RegressionDataset {
  std::vector<FeatureType> features;
  MarginalDistribution targets;
  std::map<std::string, std::string> metadata;

  RegressionDataset(){};

  RegressionDataset(const std::vector<FeatureType> &features_,
                    const MarginalDistribution &targets_)
      : features(features_), targets(targets_) {
    // If the two inputs aren't the same size they clearly aren't
    // consistent.
    assert(static_cast<int>(features.size()) ==
           static_cast<int>(targets.size()));
  }

  RegressionDataset(const std::vector<FeatureType> &features_,
                    const Eigen::VectorXd &targets_)
      : RegressionDataset(features_, MarginalDistribution(targets_)) {}

  bool operator==(const RegressionDataset &other) const {
    return (features == other.features && targets == other.targets &&
            metadata == other.metadata);
  }

  template <class Archive>
  typename std::enable_if<valid_in_out_serializer<FeatureType, Archive>::value,
                          void>::type
  serialize(Archive &archive) {
    archive(cereal::make_nvp("features", features));
    archive(cereal::make_nvp("targets", targets));
    archive(cereal::make_nvp("metadata", metadata));
  }

  template <class Archive>
  typename std::enable_if<!valid_in_out_serializer<FeatureType, Archive>::value,
                          void>::type
  serialize(Archive &) {
    static_assert(delay_static_assert<Archive>::value,
                  "In order to serialize a RegressionDataset the corresponding "
                  "FeatureType must be serializable.");
  }
};

} // namespace albatross

#endif
