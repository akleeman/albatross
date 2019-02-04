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

#include "core/model.h"
#include "covariance_functions/covariance_function.h"
#include "models/gp.h"

#include "test_utils.h"
#include <gtest/gtest.h>

namespace albatross {

/*
 * Simply makes sure that a BaseModel that should be able to
 * make perfect predictions compiles and runs as expected.
 */
TEST(test_inspect, test_linear_example) {
  double a = 5.;
  double b = 1.;
  auto dataset = make_toy_linear_data(a, b);

  Polynomial<1> linear;

  auto model = gp_from_covariance<double>(linear);
  model.fit(dataset);

  const auto polynomial_states = linear.get_state_space_representation();
  const auto state_preds = model.inspect(polynomial_states);

  // Make sure our estimates of a and b are within a couple standard deviations
  // of the truth;
  EXPECT_LE(fabs(a - state_preds.mean[0]),
            2 * sqrt(state_preds.get_diagonal(0)));
  EXPECT_LE(fabs(b - state_preds.mean[1]),
            2 * sqrt(state_preds.get_diagonal(1)));
}

/*
 * Simply makes sure that a BaseModel that should be able to
 * make perfect predictions compiles and runs as expected.
 */
TEST(test_inspect, test_heteroscedastic_linear_example) {
  double a = 5.;
  double b = 1.;
  auto dataset = make_heteroscedastic_toy_linear_data(a, b);

  Polynomial<1> linear;

  auto model = gp_from_covariance<double>(linear);
  model.fit(dataset);

  const auto polynomial_states = linear.get_state_space_representation();
  const auto state_preds = model.inspect(polynomial_states);

  // Make sure our estimates of a and b are within a standard deviation of the
  // truth;
  EXPECT_LE(fabs(a - state_preds.mean[0]),
            2 * sqrt(state_preds.get_diagonal(0)));
  EXPECT_LE(fabs(b - state_preds.mean[1]),
            2 * sqrt(state_preds.get_diagonal(1)));
}

double test_scaling(const double x) { return sin(x); }

class TestScalingFunction : public ScalingFunction {
public:
  std::string get_name() const override { return "scaling_function"; }

  double call_impl_(const double &x) const { return test_scaling(x); }
};

static inline auto make_scaled_data(const double a = 5., const double b = 1.,
                                    const double sigma = 0.1,
                                    const std::size_t n = 10) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(3);
  std::normal_distribution<> d{0., sigma};
  std::vector<double> features;
  Eigen::VectorXd mean(n);
  Eigen::VectorXd variance = sigma * sigma * Eigen::VectorXd::Ones(n);

  for (std::size_t i = 0; i < n; i++) {
    double x = static_cast<double>(i);
    features.push_back(x);
    mean[i] = test_scaling(x) * (a + x * b) + d(gen);
  }

  const MarginalDistribution targets(mean, variance.asDiagonal());
  return RegressionDataset<double>(features, targets);
}

/*
 * Simply makes sure that a BaseModel that should be able to
 * make perfect predictions compiles and runs as expected.
 */
TEST(test_inspect, test_with_scaling_function) {
  double a = 5.;
  double b = 1.;
  auto dataset = make_scaled_data(a, b);

  TestScalingFunction scaling;
  ScalingTerm<TestScalingFunction> scaling_term;

  Polynomial<1> linear;

  auto cov = scaling_term * linear;
  auto model = gp_from_covariance<double>(cov);
  model.fit(dataset);

  const auto polynomial_states = linear.get_state_space_representation();
  const auto state_preds = model.inspect(polynomial_states);

  // Make sure our estimates of a and b are within a standard deviation of the
  // truth;
  EXPECT_LE(fabs(a - state_preds.mean[0]),
            2 * sqrt(state_preds.get_diagonal(0)));
  EXPECT_LE(fabs(b - state_preds.mean[1]),
            2 * sqrt(state_preds.get_diagonal(1)));
}

} // namespace albatross
