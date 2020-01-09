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

#ifndef INCLUDE_ALBATROSS_SRC_MODELS_MCMC_HPP_
#define INCLUDE_ALBATROSS_SRC_MODELS_MCMC_HPP_

namespace albatross {

template <typename ModelType, typename FeatureType> struct MCMCFit {};

template <typename ModelType, typename FeatureType>
struct Fit<MCMCFit<ModelType, FeatureType>> {
  using FitModelType = typename fit_model_type<ModelType, FeatureType>::type;

  std::vector<FitModelType> fit_models;
};

struct StatusCallback {
  void operator()(std::size_t i, const EnsembleSamplerState &ensembles) {
    double mean = 0.;
    for (const auto &state : ensembles) {
      mean += state.log_prob / ensembles.size();
    }
    std::cout << "Iteration: " << i << "  :  " << mean << std::endl;
  };
};

template <typename ModelType> class MCMC : public ModelBase<MCMC<ModelType>> {

public:
  MCMC(const ModelType &model, std::size_t n_samples, std::size_t iterations,
       std::default_random_engine *gen)
      : model_(model), n_samples_(n_samples), iterations_(iterations),
        gen_(gen) {}

  template <typename FeatureType>
  Fit<MCMCFit<ModelType, FeatureType>>
  _fit_impl(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) const {

    std::normal_distribution<double> jitter_distribution(0., 0.1);

    auto callback = get_csv_writing_callback(model_, "./samples.csv");

    const RegressionDataset<FeatureType> dataset(features, targets);
    const std::size_t walkers = model_.get_params().size() * 3;
    const auto samples = ensemble_sampler(model_, dataset, walkers, iterations_,
                                          *gen_, callback);

    const auto final_ensemble = samples[samples.size() - 1];

    Fit<MCMCFit<ModelType, FeatureType>> fit;
    for (const auto &state : final_ensemble) {
      ModelType m(model_);
      m.set_tunable_params_values(state.params);
      fit.fit_models.push_back(m.fit(features, targets));
    }

    return fit;
  };

  template <typename FeatureType, typename FitFeaturetype>
  MarginalDistribution
  _predict_impl(const std::vector<FeatureType> &features,
                const Fit<MCMCFit<ModelType, FitFeaturetype>> &fit,
                PredictTypeIdentity<MarginalDistribution> &&) const {

    std::vector<MarginalDistribution> marginals;

    for (const auto &m : fit.fit_models) {
      marginals.emplace_back(m.predict(features).marginal());
    }

    assert(marginals.size() > 0);
    MarginalDistribution output(marginals[0]);

    const auto k = marginals.size();
    const auto n = marginals[0].mean.size();
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(n);
    for (std::size_t i = 0; i < features.size(); ++i) {
      double expected_mean = 0.;
      double expected_variance = 0.;
      for (std::size_t j = 0; j < k; ++j) {
        expected_mean += marginals[j].mean[i] / k;
        expected_variance += marginals[j].get_diagonal(i) / k;
      }

      double variance_of_mean = 0.;
      for (std::size_t j = 0; j < k; ++j) {
        variance_of_mean +=
            pow(marginals[j].mean[i] - expected_mean, 2) / (k - 1);
      }

      output.mean[i] = expected_mean;
      output.covariance.diagonal()[i] = expected_variance + variance_of_mean;
    }

    return output;
  }

private:
  ModelType model_;
  std::size_t n_samples_;
  std::size_t iterations_;
  std::default_random_engine *gen_;
};

template <typename ModelType>
auto mcmc_model(const ModelType &model, std::size_t n_samples,
                std::size_t iterations, std::default_random_engine *gen) {
  return MCMC<ModelType>(model, n_samples, iterations, gen);
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_MODELS_MCMC_HPP_ */
