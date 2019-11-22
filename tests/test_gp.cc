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

#include <albatross/src/utils/random_utils.hpp>

namespace albatross {

/*
 * In what follows we create a small problem which contains unobservable
 * components.  The model consists of a constant term for each nearest
 * integer, and another constant term for all values.  Ie, measurements
 * from the same integer wide bin will share a bias and all measurements
 * will share another bias.  The result is that the model can't
 * differentiate between the global bias and the average of all integer
 * biases (if you add 1 to the global bias and subtract 1 from all interval
 * biases you end up with the same measurements).  This is handled
 * properly by the direct Gaussian process, but if you first make a
 * prediction of each of the biases, then try to use that prediction to
 * make a new model you end up dealing with a low rank system of
 * equations which if not handled properly can lead to very large
 * errors.  This simply makes sure those errors are properly dealt with.
 */

enum InducingFeatureType { ConstantEverywhereType, ConstantPerIntervalType };

struct ConstantEverywhereFeature {};

struct ConstantPerIntervalFeature {
  ConstantPerIntervalFeature() : location(){};
  explicit ConstantPerIntervalFeature(const long &location_)
      : location(location_){};
  long location;
};

using InducingFeature =
    variant<ConstantEverywhereFeature, ConstantPerIntervalFeature>;

std::vector<InducingFeature>
create_inducing_points(const std::vector<double> &features) {

  std::vector<InducingFeature> inducing_points;
  double min = *std::min_element(features.begin(), features.end());
  double max = *std::max_element(features.begin(), features.end());

  ConstantEverywhereFeature everywhere;
  inducing_points.emplace_back(everywhere);

  long interval = lround(min);
  while (interval <= lround(max)) {
    inducing_points.emplace_back(ConstantPerIntervalFeature(interval));
    interval += 1;
  }

  return inducing_points;
}

class ConstantEverywhere : public CovarianceFunction<ConstantEverywhere> {
public:
  ConstantEverywhere(){};
  ~ConstantEverywhere(){};

  double variance = 0.1;

  /*
   * This will create a covariance matrix that looks like,
   *     sigma_mean^2 * ones(m, n)
   * which is saying all observations are perfectly correlated,
   * so you can move one if you move the rest the same amount.
   */
  double _call_impl(const double &x, const double &y) const { return variance; }

  double _call_impl(const ConstantEverywhereFeature &x, const double &y) const {
    return variance;
  }

  double _call_impl(const ConstantEverywhereFeature &x,
                    const ConstantEverywhereFeature &y) const {
    return variance;
  }
};

class ConstantPerInterval : public CovarianceFunction<ConstantPerInterval> {
public:
  ConstantPerInterval(){};
  ~ConstantPerInterval(){};

  double variance = 5.;

  /*
   * This will create a covariance matrix that looks like,
   *     sigma_mean^2 * ones(m, n)
   * which is saying all observations are perfectly correlated,
   * so you can move one if you move the rest the same amount.
   */
  double _call_impl(const double &x, const double &y) const {
    if (lround(x) == lround(y)) {
      return variance;
    } else {
      return 0.;
    }
  }

  double _call_impl(const ConstantPerIntervalFeature &x,
                    const double &y) const {
    if (x.location == lround(y)) {
      return variance;
    } else {
      return 0.;
    }
  }

  double _call_impl(const ConstantPerIntervalFeature &x,
                    const ConstantPerIntervalFeature &y) const {
    if (x.location == y.location) {
      return variance;
    } else {
      return 0.;
    }
  }
};

RegressionDataset<double> test_unobservable_dataset() {
  Eigen::Index k = 10;
  Eigen::VectorXd mean = 3.14159 * Eigen::VectorXd::Ones(k);
  Eigen::VectorXd variance = 0.1 * Eigen::VectorXd::Ones(k);
  MarginalDistribution targets(mean, variance.asDiagonal());

  std::vector<double> train_features;
  for (Eigen::Index i = 0; i < k; ++i) {
    train_features.push_back(static_cast<double>(i) * 0.3);
  }

  RegressionDataset<double> dataset(train_features, targets);
  return dataset;
}

auto test_unobservable_model() {
  ConstantEverywhere constant;
  ConstantPerInterval per_interval;
  // First we fit a model directly to the training data and use
  // that to get a prediction of the inducing points.
  auto model = gp_from_covariance(constant + per_interval, "unobservable");
  return model;
}

TEST(test_gp, test_update_model_trait) {
  const auto dataset = test_unobservable_dataset();

  auto model = test_unobservable_model();

  using FitModelType = typename fit_model_type<decltype(model), double>::type;
  using UpdatedFitType = typename updated_fit_type<FitModelType, int>::type;
  using ExpectedType =
      FitModel<decltype(model),
               Fit<GPFit<BlockSymmetric<Eigen::SerializableLDLT>,
                         variant<double, int>>>>;

  EXPECT_TRUE(bool(std::is_same<UpdatedFitType, ExpectedType>::value));
}

TEST(test_gp, test_conditionally_independent_update) {

  Eigen::VectorXd m_a(5);
  m_a << -0.23815646, 0.77162991, 0.63831216, -0.49488014, -0.08042518;

  Eigen::MatrixXd E_a(5, 5);
  E_a.row(0) << 0.09412242, -0.05731543, -0.06418745, 0.06777603, 0.01962483;
  E_a.row(1) << -0.05731543, 0.68019083, 0.1327259, -0.1612869, -0.26964463;
  E_a.row(2) << -0.06418745, 0.1327259, 0.44875809, -0.27393564, 0.17234932;
  E_a.row(3) << 0.06777603, -0.1612869, -0.27393564, 0.18412076, -0.05782831;
  E_a.row(4) << 0.01962483, -0.26964463, 0.17234932, -0.05782831, 0.23420138;

  Eigen::VectorXd m_b(5);
  m_b << 0.88905815, 0.4911579, 0.3107074, -0.52662019, -0.02583927;

  Eigen::MatrixXd E_b(5, 5);
  E_b.row(0) << 0.26633483, 0.09568965, -0.16568656, 0.00831657, -0.12015643;
  E_b.row(1) << 0.09568965, 0.11533427, 0.01014716, -0.06732859, -0.06957063;
  E_b.row(2) << -0.16568656, 0.01014716, 0.51135832, -0.26282571, 0.26183115;
  E_b.row(3) << 0.00831657, -0.06732859, -0.26282571, 0.17290452, -0.09956082;
  E_b.row(4) << -0.12015643, -0.06957063, 0.26183115, -0.09956082, 0.18919072;

  Eigen::MatrixXd S_zz(5, 5);
  S_zz.row(0) << 0.50038084, -0.0027054, -0.21695976, 0.19367812, -0.01512259;
  S_zz.row(1) << -0.0027054, 0.91975717, 0.07426668, -0.01012945, -0.30699883;
  S_zz.row(2) << -0.21695976, 0.07426668, 0.6843842, -0.33736966, 0.25493624;
  S_zz.row(3) << 0.19367812, -0.01012945, -0.33736966, 0.59627826, 0.06192099;
  S_zz.row(4) << -0.01512259, -0.30699883, 0.25493624, 0.06192099, 0.45977261;

  const auto actual = merge(m_a, E_a, m_b, E_b, S_zz);

  Eigen::VectorXd expected_mean(5);
  expected_mean << 0.66554097, 0.80310721, 0.34832714, -0.70518971, -0.23896764;
  EXPECT_LE((actual.mean - expected_mean).norm(), 1e-5);

  Eigen::MatrixXd expected_cov(5, 5);
  expected_cov.row(0) << 0.18476623, 0.02672488, -0.03349548, 0.14801588,
      0.04311095;
  expected_cov.row(1) << 0.02672465, 0.22749397, -0.02329003, 0.14191864,
      -0.0107679;
  expected_cov.row(2) << -0.03349546, -0.02329307, 0.11873036, -0.03912292,
      0.00496675;
  expected_cov.row(3) << 0.14801585, 0.14192053, -0.03912261, 0.38109587,
      0.12980902;
  expected_cov.row(4) << 0.04311105, -0.01076949, 0.00496555, 0.1298096,
      0.16257862;

  EXPECT_LE((actual.covariance - expected_cov).norm(), 1e-3);
}

TEST(test_gp, test_conditionally_independent_update_gp) {

  const auto inducing_points =
      create_inducing_points(test_unobservable_dataset().features);

  ConstantEverywhere constant;
  ConstantPerInterval per_interval;
  IndependentNoise<double> meas_noise;

  const auto cov = constant + per_interval + meas_noise;

  // First we fit a model directly to the training data and use
  // that to get a prediction of the inducing points.
  auto model = gp_from_covariance(cov, "unobservable");

  const Eigen::MatrixXd inducing_prior = cov(inducing_points);

  std::default_random_engine gen(2012);
  const Eigen::VectorXd truth = random_multivariate_normal(inducing_prior, gen);

  auto random_dataset = [&]() {
    std::vector<double> random_features(50);
    for (auto &d : random_features) {
      d = std::uniform_real_distribution<double>(-1., 4.)(gen);
    }

    const Eigen::MatrixXd cross = cov(random_features, inducing_points);
    const Eigen::MatrixXd prior = cov(random_features);

    const Eigen::VectorXd posterior_mean =
        cross * inducing_prior.ldlt().solve(truth);
    const Eigen::MatrixXd posterior_cov =
        prior - cross * inducing_prior.ldlt().solve(cross.transpose());

    const auto noise_sample = random_multivariate_normal(posterior_cov, gen);
    const MarginalDistribution random_targets(posterior_mean + noise_sample);

    return RegressionDataset<double>(random_features, random_targets);
  };

  const auto dataset_a = random_dataset();
  const auto dataset_b = random_dataset();
  const auto dataset_ab = concatenate_datasets(dataset_a, dataset_b);

  const auto fit_model_a = model.fit(dataset_a);
  const auto fit_model_b = model.fit(dataset_b);
  const auto fit_model_ab = model.fit(dataset_ab);

  const auto inducing_pred_a = fit_model_a.predict(inducing_points).joint();
  const auto inducing_pred_b = fit_model_b.predict(inducing_points).joint();
  const auto inducing_pred_ab = fit_model_ab.predict(inducing_points).joint();

  std::cout << "=====a      " << inducing_pred_a.mean.transpose() << std::endl;
  std::cout << inducing_pred_a.covariance << std::endl;
  std::cout << "=====b      " << inducing_pred_b.mean.transpose() << std::endl;
  std::cout << inducing_pred_b.covariance << std::endl;
  std::cout << "=====ab     " << inducing_pred_ab.mean.transpose() << std::endl;
  std::cout << inducing_pred_ab.covariance << std::endl;

  const auto fit_ssr_a =
      model.fit_from_prediction(inducing_points, inducing_pred_a);
  const auto fit_ssr_b =
      model.fit_from_prediction(inducing_points, inducing_pred_b);

  const auto merged = merge(fit_ssr_a, fit_ssr_b);

  std::cout << "=====merged " << merged.mean.transpose() << std::endl;
  std::cout << merged.covariance << std::endl;
  std::cout << "=====true   " << truth.transpose() << std::endl;
}

TEST(test_gp, test_update_model_same_types) {
  const auto dataset = test_unobservable_dataset();

  std::vector<std::size_t> train_inds = {0, 1, 3, 4, 6, 7, 8, 9};
  std::vector<std::size_t> test_inds = {2, 5};

  const auto train = albatross::subset(dataset, train_inds);
  const auto test = albatross::subset(dataset, test_inds);

  std::vector<std::size_t> first_inds = {0, 1, 2, 3, 5, 7};
  std::vector<std::size_t> second_inds = {4, 6};
  const auto first = albatross::subset(train, first_inds);
  const auto second = albatross::subset(train, second_inds);

  const auto model = test_unobservable_model();

  const auto full_model = model.fit(train);
  const auto full_pred = full_model.predict(test.features).joint();

  const auto first_model = model.fit(first);
  const auto split_model = update(first_model, second);
  const auto split_pred = split_model.predict(test.features).joint();

  // Make sure the fit feature type is a double
  const auto split_fit = split_model.get_fit();
  bool is_double =
      std::is_same<typename decltype(split_fit)::Feature, double>::value;
  EXPECT_TRUE(is_double);

  // Make sure a partial fit, followed by update is the same as a full fit
  EXPECT_TRUE(split_pred.mean.isApprox(full_pred.mean));
  EXPECT_LE((split_pred.covariance - full_pred.covariance).norm(), 1e-6);

  // Make sure a partial fit is not the same as a full fit
  const auto first_pred = first_model.predict(test.features).joint();
  EXPECT_FALSE(split_pred.mean.isApprox(first_pred.mean));
  EXPECT_GE((split_pred.covariance - first_pred.covariance).norm(), 1e-6);
}

TEST(test_gp, test_update_model_different_types) {
  const auto dataset = test_unobservable_dataset();

  const auto model = test_unobservable_model();
  const auto fit_model = model.fit(dataset);

  const auto inducing_points = create_inducing_points(dataset.features);
  MarginalDistribution inducing_prediction =
      fit_model.predict(inducing_points).marginal();

  inducing_prediction.covariance =
      (1e-4 * Eigen::VectorXd::Ones(inducing_prediction.mean.size()))
          .asDiagonal();

  RegressionDataset<InducingFeature> inducing_dataset(inducing_points,
                                                      inducing_prediction);
  const auto new_fit_model = update(fit_model, inducing_dataset);

  ConstantPerInterval cov;

  // Make sure the new fit with constrained inducing points reproduces
  // the prediction of the constraint
  const auto new_pred = new_fit_model.predict(inducing_points).joint();
  EXPECT_LE((new_pred.mean - inducing_prediction.mean).norm(), 0.01);
  // Without changing the prediction of the training features much
  const auto train_pred = new_fit_model.predict(dataset.features).marginal();
  EXPECT_LE((train_pred.mean - dataset.targets.mean).norm(), 0.1);

  MarginalDistribution perturbed_inducing_targets(inducing_prediction);
  perturbed_inducing_targets.mean +=
      Eigen::VectorXd::Random(perturbed_inducing_targets.mean.size());

  RegressionDataset<InducingFeature> perturbed_dataset(
      inducing_points, perturbed_inducing_targets);
  const auto new_perturbed_model = update(fit_model, perturbed_dataset);
  const auto perturbed_inducing_pred =
      new_perturbed_model.predict(inducing_points).marginal();
  const auto perturbed_train_pred =
      new_perturbed_model.predict(dataset.features).marginal();

  // Make sure constraining to a different value changes the results.
  EXPECT_GE((perturbed_inducing_pred.mean - new_pred.mean).norm(), 1.);
  EXPECT_GE((perturbed_train_pred.mean - train_pred.mean).norm(), 1.);
}

TEST(test_gp, test_model_from_different_datasets) {
  const auto unobservable_dataset = test_unobservable_dataset();

  const auto model = test_unobservable_model();

  const auto fit_model = model.fit(unobservable_dataset);
  const auto inducing_points =
      create_inducing_points(unobservable_dataset.features);
  MarginalDistribution inducing_prediction =
      fit_model.predict(inducing_points).marginal();

  // Then we create a new model in which the inducing points are
  // constrained to be the same as the previous prediction.
  inducing_prediction.covariance =
      1e-12 * Eigen::VectorXd::Ones(inducing_prediction.size()).asDiagonal();
  RegressionDataset<double> dataset(unobservable_dataset);
  RegressionDataset<InducingFeature> inducing_dataset(inducing_points,
                                                      inducing_prediction);
  const auto fit_again = model.fit(dataset, inducing_dataset);

  // Then we can make sure that the subsequent constrained predictions are
  // consistent
  const auto pred = fit_again.predict(inducing_points).joint();
  EXPECT_TRUE(inducing_prediction.mean.isApprox(pred.mean));

  const auto train_pred =
      fit_model.predict(unobservable_dataset.features).joint();
  const auto train_pred_again =
      fit_again.predict(unobservable_dataset.features).joint();
  EXPECT_TRUE(train_pred.mean.isApprox(train_pred_again.mean));

  // Now constrain the inducing points to be zero and make sure that
  // messes things up.
  inducing_dataset.targets.mean.fill(0.);
  const auto fit_zero = model.fit(dataset, inducing_dataset);
  const auto pred_zero = fit_zero.predict(inducing_points).joint();

  EXPECT_FALSE(inducing_dataset.targets.mean.isApprox(pred.mean));
  EXPECT_LT((inducing_dataset.targets.mean - pred_zero.mean).norm(), 1e-6);
}

TEST(test_gp, test_model_from_prediction_low_rank) {
  Eigen::Index k = 10;
  Eigen::VectorXd mean = 3.14159 * Eigen::VectorXd::Ones(k);
  Eigen::VectorXd variance = 0.1 * Eigen::VectorXd::Ones(k);
  MarginalDistribution targets(mean, variance.asDiagonal());

  std::vector<double> train_features;
  for (Eigen::Index i = 0; i < k; ++i) {
    train_features.push_back(static_cast<double>(i) * 0.3);
  }

  ConstantEverywhere constant;
  ConstantPerInterval per_interval;

  auto model = gp_from_covariance(constant + per_interval, "unobservable");
  const auto fit_model = model.fit(train_features, targets);

  const auto inducing_points = create_inducing_points(train_features);

  auto joint_prediction = fit_model.predict(inducing_points).joint();

  std::vector<double> perturbed_features = {50.01, 51.01, 52.01};

  const auto model_pred = fit_model.predict(perturbed_features).joint();

  auto joint_prediction_from_prediction =
      model.fit_from_prediction(inducing_points, joint_prediction)
          .predict(perturbed_features)
          .joint();

  EXPECT_TRUE(
      joint_prediction_from_prediction.mean.isApprox(model_pred.mean, 1e-12));
  EXPECT_TRUE(joint_prediction_from_prediction.covariance.isApprox(
      model_pred.covariance, 1e-8));
}

TEST(test_gp, test_unobservablemodel_with_sum_constraint) {

  const auto dataset = test_unobservable_dataset();
  const auto model = test_unobservable_model();

  const auto inducing_points = create_inducing_points(dataset.features);

  std::vector<ConstantPerIntervalFeature> interval_features;
  for (const auto &f : inducing_points) {
    if (f.is<ConstantPerIntervalFeature>()) {
      interval_features.emplace_back(f.get<ConstantPerIntervalFeature>());
    }
  }

  LinearCombination<ConstantPerIntervalFeature> sums(interval_features);

  Eigen::VectorXd mean = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd variance = 1e-5 * Eigen::VectorXd::Ones(1);
  MarginalDistribution targets(mean, variance.asDiagonal());
  RegressionDataset<LinearCombination<ConstantPerIntervalFeature>> sum_dataset(
      {sums}, targets);

  const auto both = concatenate_datasets(dataset, sum_dataset);

  const auto fit_model = model.fit(both);

  const auto pred = fit_model.predict(interval_features).joint();

  const auto ones = Eigen::VectorXd::Ones(pred.mean.size());
  EXPECT_NEAR(ones.dot(pred.mean), 0., 1e-6);
  EXPECT_NEAR(ones.dot(pred.covariance * ones), 0., 1e-5);
}

TEST(test_gp, test_unobservablemodel_with_diff_constraint) {

  const auto dataset = test_unobservable_dataset();
  const auto model = test_unobservable_model();

  const auto inducing_points = create_inducing_points(dataset.features);

  std::vector<ConstantPerIntervalFeature> interval_features;
  for (const auto &f : inducing_points) {
    if (f.is<ConstantPerIntervalFeature>()) {
      interval_features.emplace_back(f.get<ConstantPerIntervalFeature>());
    }
  }

  std::vector<ConstantPerIntervalFeature> diff_features = {
      interval_features[0], interval_features[1]};

  Eigen::Vector2d diff_coefs;
  diff_coefs << 1, -1;

  LinearCombination<ConstantPerIntervalFeature> difference(diff_features,
                                                           diff_coefs);

  Eigen::VectorXd mean = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd variance = 1e-5 * Eigen::VectorXd::Ones(1);
  MarginalDistribution targets(mean, variance.asDiagonal());
  RegressionDataset<LinearCombination<ConstantPerIntervalFeature>> diff_dataset(
      {difference}, targets);

  const auto both = concatenate_datasets(dataset, diff_dataset);

  const auto fit_model = model.fit(both);

  const auto pred = fit_model.predict(diff_features).joint();

  EXPECT_NEAR(diff_coefs.dot(pred.mean), 0., 1e-6);
  EXPECT_NEAR(diff_coefs.dot(pred.covariance * diff_coefs), 0., 1e-5);
}

} // namespace albatross
