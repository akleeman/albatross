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

#include <albatross/Tune>
#include <csv.h>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>

#include "sinc_example_utils.h"

#include <albatross/Samplers>
#include <albatross/src/models/mcmc.hpp>

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
DEFINE_string(n, "10", "number of training points to use.");
DEFINE_string(mode, "radial", "which modelling approach to use.");
DEFINE_bool(tune, false, "a flag indication parameters should be tuned first.");

using albatross::get_tuner;
using albatross::ParameterStore;
using albatross::RegressionDataset;

template <typename ModelType>
void run_model(ModelType &model, RegressionDataset<double> &data, double low,
               double high) {

  std::cout << pretty_param_details(model.get_params()) << std::endl;
  const auto fit_model = model.fit(data.features, data.targets);

  /*
   * Make predictions at a bunch of locations which we can then
   * visualize if desired.
   */
  write_predictions_to_csv(FLAGS_output, fit_model, low, high);
}

int main(int argc, char *argv[]) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int n = std::stoi(FLAGS_n);
  const double low = -10.;
  const double high = 23.;
  const double meas_noise_sd = 1.;

  if (FLAGS_input == "") {
    FLAGS_input = "input.csv";
  }
  maybe_create_training_data(FLAGS_input, n, low, high, meas_noise_sd);

  using namespace albatross;

  std::cout << "Reading the input data." << std::endl;
  RegressionDataset<double> data = read_csv_input(FLAGS_input);

  std::cout << "Defining the model." << std::endl;

  using Noise = IndependentNoise<double>;
  Noise indep_noise(meas_noise_sd);
  indep_noise.sigma_independent_noise.prior = LogScaleUniformPrior(1e-3, 1e2);

  if (FLAGS_mode == "radial") {
    // this approach uses a squared exponential radial function to capture
    // the function we're estimating using non-parametric techniques
    const Polynomial<1> polynomial(100.);
    using SquaredExp = SquaredExponential<EuclideanDistance>;
    const SquaredExp squared_exponential(3.5, 5.7);
    auto cov = polynomial + squared_exponential + measurement_only(indep_noise);
    auto model = gp_from_covariance(cov);
    run_model(model, data, low, high);
  } else if (FLAGS_mode == "parametric") {
    // Here we assume we know the "truth" is made up of a linear trend and
    // a scaled and translated sinc function with added noise and capture
    // this all through the use of a mean function.
    const LinearMean linear;
    const SincFunction sinc;
    auto model = gp_from_covariance_and_mean(indep_noise, linear + sinc);
    run_model(model, data, low, high);
  } else if (FLAGS_mode == "parametric_mcmc") {
    LinearMean linear;
    SincFunction sinc;

    sinc.scale.value = EXAMPLE_SCALE_VALUE;
    sinc.translation = EXAMPLE_TRANSLATION_VALUE;
    linear.offset.value = EXAMPLE_CONSTANT_VALUE;
    linear.slope.value = EXAMPLE_SLOPE_VALUE;

    auto model = gp_from_covariance_and_mean(indep_noise, linear + sinc);

    std::default_random_engine gen(2012);
    auto mcmc = mcmc_model(model, 32, 1500, &gen);
    run_model(mcmc, data, low, high);

  } else if (FLAGS_mode == "radial_mcmc") {
    const Polynomial<1> polynomial(100.);
    using SquaredExp = SquaredExponential<EuclideanDistance>;
    const SquaredExp squared_exponential(3.5, 5.7);
    auto cov = polynomial + squared_exponential + measurement_only(indep_noise);

    auto model = gp_from_covariance(cov);

    std::map<std::string, double> params = {
        {"sigma_independent_noise", 0.941543},
        {"sigma_polynomial_0", 500.99},
        {"sigma_polynomial_1", 0.637928},
        {"sigma_squared_exponential", 6.16107},
        {"squared_exponential_length_scale", 4.62143},
    };
    model.set_param_values(params);

    std::default_random_engine gen(2012);
    auto mcmc = mcmc_model(model, 32, 500, &gen);
    run_model(mcmc, data, low, high);

  } else if (FLAGS_mode == "linear_radial_mcmc") {

    const LinearMean linear;
    using SquaredExp = SquaredExponential<EuclideanDistance>;
    const SquaredExp squared_exponential(3.5, 5.7);
    auto model =
        gp_from_covariance_and_mean(indep_noise + squared_exponential, linear);

    std::default_random_engine gen(2012);
    auto mcmc = mcmc_model(model, 50, 500, &gen);
    run_model(mcmc, data, low, high);
  }
}
