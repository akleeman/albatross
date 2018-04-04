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

#include "gflags/gflags.h"
#include "example_utils.h"

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
DEFINE_string(n, "10", "number of training points to use.");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int n = std::stoi(FLAGS_n);
  const double low = -3.;
  const double high = 13.;
  const double meas_noise = 1.;

  maybe_create_training_data(FLAGS_input, n, low, high, meas_noise);

  auto data = read_csv_input(FLAGS_input);

  using namespace albatross;

  using Noise = IndependentNoise<double>;
  using SqrExp = SquaredExponential<ScalarDistance>;

  const auto constant_term = Constant(100.);
  CovarianceFunction<Constant> constant = {constant_term};
  CovarianceFunction<SlopeTerm> slope = {SlopeTerm(100.)};
  CovarianceFunction<Noise> noise = {Noise(meas_noise)};
  CovarianceFunction<SqrExp> sqrexp = {SqrExp(2., 5.)};



  auto linear_model = constant + slope + noise + sqrexp;

  std::cout << "Using Model:" << std::endl;
  std::cout << linear_model.to_string() << std::endl;

  auto model = gp_from_covariance(linear_model);

  model.fit(data);

  const auto constant_state = constant_term.get_state_space_representation(data.features);

  auto posterior_state = model.inspect(constant_state);
  std::cout << "The posterior estimate of the constant term is: ";
  std::cout << posterior_state.mean << " +/- " << posterior_state.covariance << std::endl;
}
