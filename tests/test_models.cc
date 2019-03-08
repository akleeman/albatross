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

#include <gtest/gtest.h>

#include "test_utils.h"

#include "GP"
#include "models/least_squares.h"

namespace albatross {

auto make_simple_covariance_function() {
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
//
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

// typedef ::testing::Types<MakeLinearRegression, MakeGaussianProcess,
//                         MakeAdaptedGaussianProcess>
typedef ::testing::Types<MakeLinearRegression> ModelCreators;
TYPED_TEST_CASE(RegressionModelTester, ModelCreators);

Eigen::Index silly_function_to_increment_stack_pointer() {
  Eigen::VectorXd x(10);
  return x.size();
}

TYPED_TEST(RegressionModelTester, performs_reasonably_on_linear_data) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  const auto fit_model = model.get_fit_model(dataset.features, dataset.targets);
  const auto pred = fit_model.get_prediction(dataset.features);
  const auto pred_mean = pred.mean();

  double rmse = sqrt((pred_mean - dataset.targets.mean).norm());
  EXPECT_LE(rmse, 0.5);
}

TYPED_TEST(RegressionModelTester, test_predict_variants) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  const auto fit_model = model.get_fit_model(dataset.features, dataset.targets);
  silly_function_to_increment_stack_pointer();
  const auto pred = fit_model.get_prediction(dataset.features);
  silly_function_to_increment_stack_pointer();

  const Eigen::VectorXd pred_mean = pred.mean();

  const MarginalDistribution marginal = pred.marginal();
  EXPECT_LE((pred_mean - marginal.mean).norm(), 1e-8);

  const JointDistribution joint = pred.joint();
  EXPECT_LE((pred_mean - joint.mean).norm(), 1e-8);
}

///*
// * Here we build two different datasets.  Each dataset consists of targets
// * which have been distorted by non-constant noise (heteroscedastic), we then
// * perform cross-validated evaluation of a GaussianProcess which takes that
// * noise into  account, and one which is agnostic of the added noise and
// assert
// * that taking noise into account improves the model.
// */
// TEST(test_models, test_with_target_distribution) {
//  auto dataset = make_heteroscedastic_toy_linear_data();
//
//  auto folds = leave_one_out(dataset);
//  auto model = MakeGaussianProcess().get_model();
//  EvaluationMetric<Eigen::VectorXd> rmse =
//      evaluation_metrics::root_mean_square_error;
//  auto scores = cross_validated_scores(rmse, folds, model.get());
//  RegressionDataset<double> dataset_without_variance(dataset.features,
//                                                     dataset.targets.mean);
//  auto folds_without_variance = leave_one_out(dataset_without_variance);
//
//  auto scores_without_variance =
//      cross_validated_scores(rmse, folds_without_variance, model.get());
//
//  EXPECT_LE(scores.mean(), scores_without_variance.mean());
//}
//
// TYPED_TEST(RegressionModelTester, cross_validation_variants) {
//  auto dataset = this->creator.get_dataset();
//  auto folds = leave_one_out(dataset);
//  auto model = this->creator.create();
//  EvaluationMetric<Eigen::VectorXd> rmse =
//      evaluation_metrics::root_mean_square_error;
//  auto cv_scores = cross_validated_scores(rmse, folds, model.get());
//
//  auto loo_indexers = leave_one_out_indexer(dataset);
//  auto loo_predictions =
//      model->template cross_validated_predictions<MarginalDistribution>(
//          dataset, loo_indexers);
//
//  auto cv_fast_scores =
//      cross_validated_scores(rmse, dataset, loo_indexers, model.get());
//
//  // Here we make sure the cross validated mean absolute error is reasonable.
//  // Note that because we are running leave one out cross validation, the
//  // RMSE for each fold is just the absolute value of the error.
//  EXPECT_LE(cv_scores.mean(), 0.1);
//}
//
// class MakeLargeGaussianProcess : public AbstractTestModel<double> {
// public:
//  std::unique_ptr<RegressionModel<double>> create() const override {
//    auto covariance = make_simple_covariance_function();
//    return gp_pointer_from_covariance<double>(covariance);
//  }
//
//  RegressionDataset<double> get_dataset() const override {
//    return make_toy_linear_data(5., 1., 0.1, 100);
//  }
//};
//
// class MakeLargeAdaptedGaussianProcess
//    : public AbstractTestModel<AdaptedFeature> {
// public:
//  std::unique_ptr<RegressionModel<AdaptedFeature>> create() const override {
//    auto covariance = make_simple_covariance_function();
//    auto gp = gp_from_covariance<double>(covariance);
//    return std::make_unique<AdaptedExample<decltype(gp)>>(gp);
//  }
//
//  RegressionDataset<AdaptedFeature> get_dataset() const override {
//    return make_adapted_toy_linear_data(5., 1., 0.1, 100);
//  }
//};
//
// template <typename ModelCreator>
// class SpecializedRegressionModelTester : public ::testing::Test {
// public:
//  ModelCreator creator;
//};
//
// typedef ::testing::Types<MakeLargeGaussianProcess,
//                         MakeLargeAdaptedGaussianProcess>
//    SpecializedModelCreators;
// TYPED_TEST_CASE(SpecializedRegressionModelTester, SpecializedModelCreators);
//
// TYPED_TEST(SpecializedRegressionModelTester,
//           test_uses_specialized_cross_validation_functions) {
//  auto dataset = this->creator.get_dataset();
//  auto model = this->creator.create();
//
//  auto loo_indexers = leave_one_out_indexer(dataset);
//  EvaluationMetric<Eigen::VectorXd> rmse =
//      evaluation_metrics::root_mean_square_error;
//
//  // time the computation of RMSE using the fast LOO variant.
//  using namespace std::chrono;
//  high_resolution_clock::time_point start = high_resolution_clock::now();
//  auto cv_fast_scores =
//      cross_validated_scores(rmse, dataset, loo_indexers, model.get());
//  high_resolution_clock::time_point end = high_resolution_clock::now();
//  auto fast_duration = duration_cast<microseconds>(end - start).count();
//
//  // time RMSE using the default method.
//  const auto folds = folds_from_fold_indexer(dataset, loo_indexers);
//  start = high_resolution_clock::now();
//  const auto cv_slow_scores = cross_validated_scores(rmse, folds,
//  model.get());
//  end = high_resolution_clock::now();
//  auto slow_duration = duration_cast<microseconds>(end - start).count();
//  // Make sure the faster variant is actually faster and that the results
//  // are the same.
//  EXPECT_LT(fast_duration, slow_duration);
//  EXPECT_NEAR((cv_fast_scores - cv_slow_scores).norm(), 0., 1e-8);
//}

} // namespace albatross
