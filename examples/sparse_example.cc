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

#include <albatross/SparseGP>

#include <csv.h>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>

#define EXAMPLE_SLOPE_VALUE 0.
#define EXAMPLE_CONSTANT_VALUE 0.

#include "sinc_example_utils.h"

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
DEFINE_int32(n, 10, "number of training points to use.");
DEFINE_int32(k, 5, "number of training points to use.");
DEFINE_bool(tune, false, "a flag indication parameters should be tuned first.");

using albatross::get_tuner;
using albatross::ParameterStore;
using albatross::RegressionDataset;

template <typename ModelType>
albatross::ParameterStore tune_model(ModelType &model,
                                     RegressionDataset<double> &data) {
  /*
   * Now we tune the model by finding the hyper parameters that
   * maximize the likelihood (or minimize the negative log likelihood).
   */
  std::cout << "Tuning the model." << std::endl;

  albatross::LeaveOneOutLikelihood<> loo_nll;

  return get_tuner(model, loo_nll, data).tune();
}

bool get_interval(const double &x) {
  return x <= 5.;
}

std::vector<double> boundaries(bool x, bool y) {
  std::vector<double> boundary = {4., 5., 6.};
  return boundary;
}

int main(int argc, char *argv[]) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int n = FLAGS_n;
  const double low = -3.;
  const double high = 13.;
  const double meas_noise = 0.1;

  if (FLAGS_input == "") {
    FLAGS_input = "input.csv";
  }
  maybe_create_training_data(FLAGS_input, n, low, high, meas_noise);

  using namespace albatross;

  std::cout << "Reading the input data." << std::endl;
  RegressionDataset<double> data = read_csv_input(FLAGS_input);

  std::cout << "Defining the model." << std::endl;
  using Noise = IndependentNoise<double>;
  using SquaredExp = SquaredExponential<EuclideanDistance>;

  Noise noise(meas_noise);
  SquaredExp squared_exponential(3.5, 5.7);
  auto cov_func = noise + squared_exponential;

  std::cout << cov_func.pretty_string() << std::endl;

  const auto m = gp_from_covariance(cov_func, "example_model");

  auto fit_to_interval = [&](const auto &dataset) {
    return m.fit(dataset);
  };

  const auto fit_models = data.group_by(get_interval).apply(fit_to_interval);

  auto C_dd_solve = [&](const Eigen::MatrixXd &x) {
    Eigen::Index i = 0;
    Eigen::MatrixXd output(x.rows(), x.cols());
    for (const auto &f : fit_models.values()) {
      const auto cov = f.get_fit().train_covariance;
      const auto rhs_chunk = x.block(i, 0, cov.rows(), x.cols());
      output.block(i, 0, cov.rows(), x.cols()) = cov.solve(rhs_chunk);
      i += cov.rows();
    }
    return output;
  };

  auto get_obs_vector = [](const auto &fit_model) {
    return fit_model.predict(fit_model.get_fit().train_features).mean();
  };
  const auto obs_vectors = fit_models.apply(get_obs_vector).values();
  const auto y = concatenate(obs_vectors[0], obs_vectors[1]);

  auto get_features = [](const auto &fit_model) {
    return fit_model.get_fit().train_features;
  };

  auto cov_func_fb = [&](const auto &features, const auto &b_features) {
    Eigen::MatrixXd output = cov_func(features, b_features);

    std::cout << "2" << std::endl;
    for (Eigen::Index i = 0; i < output.rows(); ++i) {
      for (Eigen::Index j = 0; j < output.cols(); ++j) {
        if (!get_interval(features[i])) {
          output(i, j) = -output(i, j);
        }
      }
    }
    std::cout << "3" << std::endl;

    return output;
  };

  const auto both_features = fit_models.apply(get_features).values();
  const auto train_features = concatenate(both_features[0], both_features[1]);

  const auto groups = fit_models.keys();
  const auto boundary = boundaries(groups[0], groups[1]);

  const Eigen::MatrixXd C_bb = 2. * cov_func(boundary);
  auto C_db = cov_func_fb(train_features, boundary);

  std::cout << "4" << std::endl;

  const Eigen::MatrixXd C_dd_inv_C_db = C_dd_solve(C_db);

  std::cout << "5" << std::endl;

  std::cout << C_bb.rows() << " " << C_bb.cols() << std::endl;
  std::cout << C_db.rows() << " " << C_db.cols() << std::endl;
  std::cout << C_dd_inv_C_db.rows() << " " << C_dd_inv_C_db.cols() << std::endl;

  const Eigen::MatrixXd S_bb = C_bb - C_db.transpose() * C_dd_inv_C_db;
  std::cout << "0" << std::endl;
  const auto S_bb_ldlt = S_bb.ldlt();
  std::cout << "1" << std::endl;

  auto solver = [&](const auto &x) {
    Eigen::MatrixXd output = C_dd_inv_C_db.transpose() * x;
    output = S_bb_ldlt.solve(output);
    output = C_dd_inv_C_db * output;
    output += C_dd_solve(x);
    return output;
  };

  const Eigen::VectorXd new_information = solver(y);

  std::cout << "a" << std::endl;
  const Eigen::MatrixXd C_bb_inv_C_bd = C_bb.ldlt().solve(C_db.transpose());
  std::cout << "b" << std::endl;

  auto cross_matrix = [&](const auto &features) {
    // Maybe only part of this;
    Eigen::MatrixXd output = cov_func(features, train_features);

    for (Eigen::Index i = 0; i < output.rows(); ++i) {
      for (Eigen::Index j = 0; j < output.cols(); ++j) {
        if (get_interval(features[i]) != get_interval(train_features[j])) {
          output(i, j) = 0.;
        }
      }
    }

    output -= cov_func_fb(features, boundary) * C_bb_inv_C_bd;
    return output;
  };

  std::ofstream output;
  output.open(FLAGS_output);

  const std::size_t k = 161;
  auto grid_xs = uniform_points_on_line(k, low - 2., high + 2.);

  const auto cross = cross_matrix(grid_xs);
  std::cout << cross.rows() << " , " << cross.cols() << std::endl;
  std::cout << C_dd_inv_C_db.rows() << " , " << C_dd_inv_C_db.cols() << std::endl;
  const Eigen::VectorXd pred_mean = cross * new_information;

  Eigen::MatrixXd cov = cov_func(grid_xs);
  const Eigen::MatrixXd C_gb = cov_func_fb(grid_xs, boundary);
  std::cout << "c" << std::endl;
  cov = cov - C_gb * C_bb.ldlt().solve(C_gb.transpose());
  std::cout << "d" << std::endl;
  cov = cov - cross * solver(cross.transpose());

  const MarginalDistribution prediction(pred_mean, cov.diagonal());

  Eigen::VectorXd targets(static_cast<Eigen::Index>(k));
  for (std::size_t i = 0; i < k; i++) {
    targets[static_cast<Eigen::Index>(i)] = truth(grid_xs[i]);
  }

  const albatross::RegressionDataset<double> dataset(grid_xs, targets);

  albatross::write_to_csv(output, dataset, prediction);
  write_predictions_to_csv(FLAGS_output + "0", fit_models.values()[0], low, high);
  write_predictions_to_csv(FLAGS_output + "1", fit_models.values()[1], low, high);

//  LeaveOneOutGrouper loo;
//  UniformlySpacedInducingPoints strategy(FLAGS_k);
//  auto model = sparse_gp_from_covariance(cov, loo, strategy, "example");
//  //  auto model = gp_from_covariance(cov, "example");
//
//  if (FLAGS_tune) {
//    model.set_params(tune_model(model, data));
//  }
//
//  std::cout << pretty_param_details(model.get_params()) << std::endl;
//  const auto fit_model = model.fit(data);

  /*
   * Make predictions at a bunch of locations which we can then
   * visualize if desired.
   */
//  write_predictions_to_csv(FLAGS_output, fit_model, low, high);
}
