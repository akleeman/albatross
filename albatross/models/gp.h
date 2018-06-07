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

#ifndef ALBATROSS_GP_GP_H
#define ALBATROSS_GP_GP_H

#include "evaluate.h"
#include "stdio.h"
#include <functional>
#include <memory>

#include "cereal/eigen.h"
#include "core/model.h"
#include "core/serialize.h"
#include "eigen/serializable_ldlt.h"

namespace albatross {

using InspectionDistribution = PredictDistribution;

template <typename FeatureType> struct GaussianProcessFit {
  std::vector<FeatureType> train_features;
  Eigen::SerializableLDLT train_ldlt;
  Eigen::VectorXd information;

  template <typename Archive>
  // todo: enable if FeatureType is serializable
  void serialize(Archive &archive) {
    archive(cereal::make_nvp("information", information));
    archive(cereal::make_nvp("train_ldlt", train_ldlt));
    archive(cereal::make_nvp("train_features", train_features));
  }

  bool operator==(const GaussianProcessFit &other) const {
    return (train_features == other.train_features &&
            train_ldlt == other.train_ldlt && information == other.information);
  }
};

template <typename FeatureType, typename SubFeatureType = FeatureType>
using SerializableGaussianProcess =
    SerializableRegressionModel<FeatureType,
                                GaussianProcessFit<SubFeatureType>>;

template <typename FeatureType, typename CovarianceFunction>
class GaussianProcessRegression
    : public SerializableGaussianProcess<FeatureType> {
public:
  typedef GaussianProcessFit<FeatureType> FitType;
  typedef CovarianceFunction CovarianceType;

  GaussianProcessRegression()
      : covariance_function_(), model_name_(covariance_function_.get_name()){};
  GaussianProcessRegression(CovarianceFunction &covariance_function)
      : covariance_function_(covariance_function),
        model_name_(covariance_function_.get_name()){};
  /*
   * Sometimes it's nice to be able to provide a custom model name since
   * these models are generalizable.
   */
  GaussianProcessRegression(CovarianceFunction &covariance_function,
                            const std::string &model_name)
      : covariance_function_(covariance_function), model_name_(model_name){};
  GaussianProcessRegression(const std::string &model_name)
      : covariance_function_(), model_name_(model_name){};

  ~GaussianProcessRegression(){};

  std::string get_name() const override { return model_name_; };

  template <typename Archive> void save(Archive &archive) const {
    archive(cereal::base_class<SerializableRegressionModel<
                FeatureType, GaussianProcessFit<FeatureType>>>(this));
    archive(model_name_);
  }

  template <typename Archive> void load(Archive &archive) {
    archive(cereal::base_class<SerializableRegressionModel<
                FeatureType, GaussianProcessFit<FeatureType>>>(this));
    archive(model_name_);
  }

  template <typename OtherFeatureType>
  InspectionDistribution
  inspect(const std::vector<OtherFeatureType> &features) const {
    assert(this->has_been_fit());
    const auto cross_cov = asymmetric_covariance(
        covariance_function_, features, this->model_fit_.train_features);
    // Then we can use the information vector to determine the posterior
    const Eigen::VectorXd pred = cross_cov * this->model_fit_.information;
    Eigen::MatrixXd pred_cov =
        symmetric_covariance(covariance_function_, features);
    auto ldlt = this->model_fit_.train_ldlt;
    pred_cov -= cross_cov * ldlt.solve(cross_cov.transpose());
    assert(static_cast<s32>(pred.size()) == static_cast<s32>(features.size()));
    return InspectionDistribution(pred, pred_cov);
  }

  /*
   * The Gaussian Process Regression model derives its parameters from
   * the covariance functions.
   */
  ParameterStore get_params() const override {
    return covariance_function_.get_params();
  }

  void unchecked_set_param(const std::string &name,
                           const double value) override {
    covariance_function_.set_param(name, value);
  }

  std::string pretty_string() const {
    std::ostringstream ss;
    ss << get_name() << std::endl;
    ss << covariance_function_.pretty_string();
    ss << "has_been_fit: " << this->has_been_fit() << std::endl;
    return ss.str();
  }

protected:
  FitType serializable_fit_(const std::vector<FeatureType> &features,
                            const TargetDistribution &targets) const override {
    Eigen::MatrixXd cov = symmetric_covariance(covariance_function_, features);
    FitType model_fit;
    model_fit.train_features = features;
    if (targets.has_covariance()) {
      cov += targets.covariance;
    }
    model_fit.train_ldlt = Eigen::SerializableLDLT(cov.ldlt());
    // Precompute the information vector
    model_fit.information = model_fit.train_ldlt.solve(targets.mean);
    return model_fit;
  }

  PredictDistribution
  predict_(const std::vector<FeatureType> &features) const override {
    const auto cross_cov = asymmetric_covariance(
        covariance_function_, features, this->model_fit_.train_features);
    // Then we can use the information vector to determine the posterior
    const Eigen::VectorXd pred = cross_cov * this->model_fit_.information;
    // TODO: right now this is recomputing the LDLT, so is highly inefficient,
    // Ideally this would get stored inside GaussianProcessFit.
    Eigen::MatrixXd pred_cov =
        symmetric_covariance(covariance_function_, features);
    auto ldlt = this->model_fit_.train_ldlt;
    pred_cov -= cross_cov * ldlt.solve(cross_cov.transpose());
    return PredictDistribution(pred, pred_cov);
  }

private:
  CovarianceFunction covariance_function_;
  std::string model_name_;
};

/*
 * The leave one out cross validated predictions for a Gaussian Process
 * can be efficiently computed by dropping a row and column from the
 * covariance and obtaining the prediction for the dropped index.  This
 * results in,
 *
 * mean[i] = y[i] - cov^{-1} y)/cov^{-1}[i, i]
 * variance[i] = 1. / cov^{-1}[i, i]
 *
 * See section 5.4.2 Rasmussen Gaussian Processes
 */
static inline Distribution<DiagonalMatrixXd>
fast_gp_loo_cross_validated_predict(
    const Eigen::VectorXd &targets,
    const Eigen::SerializableLDLT &train_covariance_ldlt) {
  assert(targets.size() == train_covariance_ldlt.rows());
  assert(train_covariance_ldlt.rows() == train_covariance_ldlt.cols());
  Eigen::VectorXd information = train_covariance_ldlt.solve(targets);
  const auto inverse = train_covariance_ldlt.solve(
      Eigen::MatrixXd::Identity(targets.size(), targets.size()));

  Eigen::VectorXd loo_mean(targets);
  Eigen::VectorXd loo_variance(targets.size());
  for (Eigen::Index i = 0; i < targets.size(); i++) {
    loo_mean[i] -= information[i] / inverse(i, i);
    loo_variance[i] = 1. / inverse(i, i);
  }
  return Distribution<DiagonalMatrixXd>(loo_mean, loo_variance.asDiagonal());
}

template <typename FeatureType, typename SubFeatureType = FeatureType>
static inline Distribution<DiagonalMatrixXd>
fast_gp_loo_cross_validated_predict(
    const RegressionDataset<FeatureType> &dataset,
    SerializableGaussianProcess<FeatureType, SubFeatureType> *model) {
  model->fit(dataset);
  const auto model_fit = model->get_fit();
  return fast_gp_loo_cross_validated_predict(dataset.targets.mean,
                                             model_fit.train_ldlt);
}

template <typename FeatureType, typename CovFunc>
GaussianProcessRegression<FeatureType, CovFunc>
gp_from_covariance(CovFunc covariance_function) {
  return GaussianProcessRegression<FeatureType, CovFunc>(covariance_function);
};

template <typename FeatureType, typename CovFunc>
GaussianProcessRegression<FeatureType, CovFunc>
gp_from_covariance(CovFunc covariance_function, const std::string &model_name) {
  return GaussianProcessRegression<FeatureType, CovFunc>(covariance_function,
                                                         model_name);
};

template <typename FeatureType, typename CovFunc>
std::unique_ptr<GaussianProcessRegression<FeatureType, CovFunc>>
gp_pointer_from_covariance(CovFunc covariance_function) {
  return std::make_unique<GaussianProcessRegression<FeatureType, CovFunc>>(
      covariance_function);
}

template <typename FeatureType, typename CovFunc>
std::unique_ptr<GaussianProcessRegression<FeatureType, CovFunc>>
gp_pointer_from_covariance(CovFunc covariance_function,
                           const std::string &model_name) {
  return std::make_unique<GaussianProcessRegression<FeatureType, CovFunc>>(
      covariance_function, model_name);
}
} // namespace albatross

#endif
