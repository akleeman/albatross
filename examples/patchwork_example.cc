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

#include <albatross/Tune>

#include <albatross/GP>
#include <albatross/utils/eigen_utils.h>

#include <csv.h>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>

#define EXAMPLE_SLOPE_VALUE 0.
#define EXAMPLE_CONSTANT_VALUE 3.14159

#include "sinc_example_utils.h"

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
DEFINE_string(n, "10", "number of training points to use.");
DEFINE_bool(tune, false, "a flag indication parameters should be tuned first.");

using albatross::get_tuner;
using albatross::ParameterStore;
using albatross::RegressionDataset;
using albatross::MarginalDistribution;

inline MarginalDistribution concatenate_marginal_predictions(
    const std::vector<MarginalDistribution> &marginals) {

  std::vector<Eigen::VectorXd> means;
  for (const auto &m : marginals) {
    means.emplace_back(m.mean);
  }
  auto mean = albatross::vertical_stack(means);

  std::vector<Eigen::VectorXd> diags;
  for (const auto &m : marginals) {
    diags.emplace_back(m.covariance.diagonal());
  }
  auto variance = albatross::vertical_stack(diags);

  return MarginalDistribution(mean, variance.asDiagonal());
}

/*
 * Create random noisy observations to use as train data.
 */
std::vector<albatross::RegressionDataset<double>>
create_disjoint_train_datasets(const int n, const double low, const double high,
                               const double measurement_noise) {

  std::size_t k = 4;
  double window_width = (high - low) / 4.;

  std::vector<albatross::RegressionDataset<double>> output;
  for (std::size_t i = 0; i < k; ++i) {
    auto d = create_train_data(n, low + i * window_width, low + (i + 1) * window_width, measurement_noise);
    output.emplace_back(d);
  }

  return output;
}

template <typename ModelType>
auto naive_patchwork(const std::vector<std::vector<double>> &features_per_group,
                     const std::vector<albatross::JointDistribution> &predictions,
                     ModelType &model) {

  std::vector<MarginalDistribution> marginals;
  for (const auto &p : predictions) {
    marginals.emplace_back(MarginalDistribution(p.mean, p.covariance.diagonal().asDiagonal()));
  }

  auto all_preds = concatenate_marginal_predictions(marginals);

  std::vector<double> all_features;
  for (const auto &features : features_per_group) {
    for (const auto &f : features) {
      all_features.emplace_back(f);
    }
  }

  albatross::RegressionDataset<double> dataset(all_features, all_preds);
  return model.fit(dataset);
}

int main(int argc, char *argv[]) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int n = std::stoi(FLAGS_n);
  const double low = -3.;
  const double high = 13.;
  const double meas_noise = 0.1;

  if (FLAGS_input == "") {
    FLAGS_input = "input.csv";
  }

  auto datasets = create_disjoint_train_datasets(n, low, high, meas_noise);

  {
    std::ofstream train_file(FLAGS_input);
    albatross::write_to_csv(train_file, datasets);
  }

  using namespace albatross;

  std::cout << "Defining the model." << std::endl;
  using Noise = IndependentNoise<double>;
  using SquaredExp = SquaredExponential<EuclideanDistance>;

  Polynomial<1> polynomial(100.);
  Noise noise(meas_noise);
  SquaredExp squared_exponential(3.5, 5.7);
  auto cov = polynomial + noise + squared_exponential;

  std::cout << cov.pretty_string() << std::endl;

  auto model = gp_from_covariance(cov);

  std::cout << pretty_param_details(model.get_params()) << std::endl;

  /*
   * Here we produce a prediction for each chunk.  We offset each
   * prediction to represent a biased prediction that may result
   * from unobservability of a constant (for example).  We then
   * add a large constant term to the predicted distribution to
   * reflect the presence of a bias.
   */
  double offset = 0.;
  std::vector<std::vector<double>> features;
  std::vector<JointDistribution> predictions;
  for (auto &d : datasets) {
    d.targets.mean.array() += offset;
    const auto fit_model = model.fit(d);

    auto pred = fit_model.predict(d.features).joint();
    pred.covariance.array() += 5.;

    predictions.emplace_back(pred);
    features.emplace_back(d.features);

    // Increment the offset
    offset += 2.;
  }

  /*
   * It's up to the patchwork model to then stitch all these predictions
   * back together in a way that accounts for possibly offset biases.
   */
  auto patchwork_model = naive_patchwork(features, predictions, model);

  /*
   * Make predictions at a bunch of locations which we can then
   * visualize if desired.
   */
  write_predictions_to_csv(FLAGS_output, patchwork_model, low, high);
}
