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

#include <gtest/gtest.h>

#include "test_utils.h"

#include "GP"
#include "models/least_squares.h"

namespace albatross {

inline auto make_simple_covariance_function() {
  SquaredExponential<EuclideanDistance> squared_exponential(100., 100.);
  IndependentNoise<double> noise(0.1);
  return squared_exponential + noise;
}

class MakeGaussianProcess {
public:
  auto get_model() const {
    auto covariance = make_simple_covariance_function();
    return gp_from_covariance(covariance);
  }

  RegressionDataset<double> get_dataset() const {
    return make_toy_linear_data();
  }
};

template <typename CovarianceFunc>
class AdaptedGaussianProcess
    : public GaussianProcessBase<CovarianceFunc,
                                 AdaptedGaussianProcess<CovarianceFunc>> {
public:
  using Base = GaussianProcessBase<CovarianceFunc,
                                   AdaptedGaussianProcess<CovarianceFunc>>;

  template <typename FitFeatureType>
  using GPFitType = Fit<Base, FitFeatureType>;

  auto fit(const std::vector<AdaptedFeature> &features,
           const MarginalDistribution &targets) const {
    std::vector<double> converted;
    for (const auto &f : features) {
      converted.push_back(f.value);
    }
    return Base::fit(converted, targets);
  }

  template <typename FitFeatureType>
  JointDistribution predict(const std::vector<AdaptedFeature> &features,
                            const GPFitType<FitFeatureType> &gp_fit,
                            PredictTypeIdentity<JointDistribution> &&) const {
    std::vector<double> converted;
    for (const auto &f : features) {
      converted.push_back(f.value);
    }
    return Base::predict(converted, gp_fit,
                         PredictTypeIdentity<JointDistribution>());
  }
};

class MakeAdaptedGaussianProcess {
public:
  auto get_model() const {
    auto covariance = make_simple_covariance_function();
    AdaptedGaussianProcess<decltype(covariance)> gp;

    return gp;
  }

  RegressionDataset<AdaptedFeature> get_dataset() const {
    return make_adapted_toy_linear_data();
  }
};

class MakeLinearRegression {
public:
  LinearRegression get_model() const { return LinearRegression(); }

  RegressionDataset<double> get_dataset() const {
    return make_toy_linear_data();
  }
};

template <typename ModelTestCase>
class RegressionModelTester : public ::testing::Test {
public:
  ModelTestCase test_case;
};

typedef ::testing::Types<MakeLinearRegression, MakeGaussianProcess,
                         MakeAdaptedGaussianProcess>
    ExampleModels;

TYPED_TEST_CASE_P(RegressionModelTester);

enum PredictLevel { MEAN, MARGINAL, JOINT };

/*
 * This TestPredictVariants struct provides different levels of
 * testing depending on what sort of predictions are available.
 */
template <typename PredictionType, typename = void> struct TestPredictVariants {
  PredictLevel test(const PredictionType &pred) {
    const Eigen::VectorXd pred_mean = pred.mean();
    EXPECT_GT(pred_mean.size(), 0);
    std::cout << "MEAN" << std::endl;
    return PredictLevel::MEAN;
  }
};

template <typename PredictionType>
struct TestPredictVariants<
    PredictionType, std::enable_if_t<has_marginal<PredictionType>::value &&
                                     !has_joint<PredictionType>::value>> {
  PredictLevel test(const PredictionType &pred) {
    const Eigen::VectorXd pred_mean = pred.mean();
    const MarginalDistribution marginal = pred.marginal();
    EXPECT_LE((pred_mean - marginal.mean).norm(), 1e-8);
    std::cout << "MARGINAL" << std::endl;
    return PredictLevel::MARGINAL;
  }
};

template <typename PredictionType>
struct TestPredictVariants<PredictionType,
                           std::enable_if_t<has_joint<PredictionType>::value>> {
  PredictLevel test(const PredictionType &pred) {
    const Eigen::VectorXd pred_mean = pred.mean();
    const MarginalDistribution marginal = pred.marginal();
    EXPECT_LE((pred_mean - marginal.mean).norm(), 1e-8);
    const JointDistribution joint = pred.joint();
    EXPECT_LE((pred_mean - joint.mean).norm(), 1e-8);
    EXPECT_LE(
        (marginal.covariance.diagonal() - joint.covariance.diagonal()).norm(),
        1e-8);
    std::cout << "JOINT" << std::endl;
    return PredictLevel::JOINT;
  }
};

template <typename PredictionType>
void expect_predict_variants_consistent(const PredictionType &pred) {
  TestPredictVariants<PredictionType> tester;
  const auto level = tester.test(pred);
  // Just in case the traits above don't work.
  if (has_mean<PredictionType>::value) {
    EXPECT_GE(level, PredictLevel::MEAN);
  }

  if (has_marginal<PredictionType>::value) {
    EXPECT_GE(level, PredictLevel::MARGINAL);
  }

  if (has_joint<PredictionType>::value) {
    EXPECT_GE(level, PredictLevel::JOINT);
  }
}


} // namespace albatross
