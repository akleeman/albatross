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

#include "evaluate.h"
#include "gflags/gflags.h"
#include "temperature_example_utils.h"
#include "tune.h"
#include <functional>

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(predict, "", "path to csv containing prediction locations.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
DEFINE_string(thin, "1", "path where predictions will be written in csv.");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  using namespace albatross;

  std::cout << "Reading the input data." << std::endl;
  auto data = read_temperature_csv_input(FLAGS_input, std::stoi(FLAGS_thin));
  std::cout << "Using " << data.features.size() << " data points" << std::endl;

  std::cout << "Defining the model." << std::endl;
  // Measurement Noise
  using Noise = IndependentNoise<Station>;
  CovarianceFunction<Noise> noise = {Noise(2.0)};

  // A Constant temperature value
  CovarianceFunction<Constant> mean = {Constant(1.5)};

  // Scale the constant temperature value in a way that defaults
  // to colder values for higher elevations.
  using ElevationScalar = ScalingTerm<ElevationScalingFunction>;
  CovarianceFunction<ElevationScalar> elevation_scalar = {ElevationScalar()};
  auto elevation_scaled_mean = elevation_scalar * mean;

  // Radial distance is the difference in lengths of the X, Y, Z
  // locations, which translates into a difference in height so
  // this term means "station at different elevations will be less correlated"
  using RadialSqrExp = SquaredExponential<StationDistance<RadialDistance>>;
  CovarianceFunction<RadialSqrExp> radial_sqr_exp = {RadialSqrExp(15000., 2.5)};

  // The angular distance is equivalent to the great circle distance
  using AngularExp = Exponential<StationDistance<AngularDistance>>;
  CovarianceFunction<AngularExp> angular_exp = {AngularExp(9e-2, 3.5)};

  // We multiply the angular and elevation covariance terms.  To justify this
  // think of the extremes.  If two stations are really far apart, regardless
  // of their elevation they should be decorrelated.  Similarly if two stations
  // are close but are at extremely different elevations they should be
  // decorrelated.
  auto spatial_cov = angular_exp * radial_sqr_exp;

  auto covariance = elevation_scaled_mean + noise + spatial_cov;
  auto model = gp_from_covariance<Station>(covariance);

  // These parameters are that came from tuning the model to the leave
  // one out negative log likelihood.
  ParameterStore params = {
      {"elevation_scaling_center", 3965.98},
      {"elevation_scaling_factor", 0.000810492},
      {"exponential_length_scale", 28197.6},
      {"squared_exponential_length_scale", 0.0753042},
      {"sigma_constant", 1.66872},
      {"sigma_exponential", 2.07548},
      {"sigma_independent_noise", 1.8288},
      {"sigma_squared_exponential", 3.77329},
  };
  model.set_params(params);

  std::cout << "Training the model." << std::endl;
  model.fit(data);

  auto predict_features = read_temperature_csv_input(FLAGS_predict, 1).features;
  std::cout << "Going to predict at " << predict_features.size() << " locations"
            << std::endl;
  write_predictions(FLAGS_output, predict_features, model);
}
