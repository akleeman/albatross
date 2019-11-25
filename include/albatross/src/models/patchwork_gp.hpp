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

#ifndef INCLUDE_ALBATROSS_MODELS_PATCHWORK_GP_H_
#define INCLUDE_ALBATROSS_MODELS_PATCHWORK_GP_H_

namespace albatross {

template <typename FitModelType, typename GrouperFunction>
struct PatchworkGPFit {};

template <typename ModelType, typename FitType, typename GrouperFunction>
struct Fit<PatchworkGPFit<FitModel<ModelType, FitType>, GrouperFunction>> {

  using FeatureType = typename FitType::Feature;
  using GroupKey = typename details::grouper_result<GrouperFunction,
                                                    FeatureType>::type;

  Fit() {};

  template <typename MaybeGroupKey>
  Fit(const Grouped<MaybeGroupKey, FitModel<ModelType, FitType>> &fit_models_,
      GrouperFunction grouper_function_) :
      fit_models(fit_models_), grouper_function(grouper_function_) {

    static_assert(std::is_same<MaybeGroupKey, GroupKey>::value,
        "The GroupKey in fit_models_ doesn't match what the grouper function returns");
  };

  Fit(const Grouped<GroupKey, FitModel<ModelType, FitType>> &&fit_models_,
      GrouperFunction grouper_function_) : fit_models(std::move(fit_models_)),
          grouper_function(grouper_function_) {};

  Grouped<GroupKey, FitModel<ModelType, FitType>> fit_models;
  GrouperFunction grouper_function;
};

template <typename CovFunc, typename GrouperFunction,
          typename BoundaryFunction>
class PatchworkGaussianProcess
    : public GaussianProcessBase<
          CovFunc, PatchworkGaussianProcess<CovFunc, GrouperFunction,
          BoundaryFunction>> {

public:
  using Base = GaussianProcessBase<
      CovFunc, PatchworkGaussianProcess<CovFunc, GrouperFunction,
                                        BoundaryFunction>>;

  PatchworkGaussianProcess() : Base() {};
  PatchworkGaussianProcess(CovFunc &covariance_function)
      : Base(covariance_function) {};
  PatchworkGaussianProcess(CovFunc &covariance_function,
      GrouperFunction grouper_function,
      BoundaryFunction boundary_function)
      : Base(covariance_function), grouper_function_(grouper_function), boundary_function_(boundary_function) {
  };

  template <typename FitModelType, typename GroupKey>
  auto from_fit_models(const Grouped<GroupKey, FitModelType> &fit_models) const {
    using PatchworkFitType = Fit<PatchworkGPFit<FitModelType, GrouperFunction>>;
    return FitModel<PatchworkGaussianProcess, PatchworkFitType>(*this, PatchworkFitType(fit_models, grouper_function_));
  };

  template <typename FeatureType, typename FitModelType>
  JointDistribution _predict_impl(
      const std::vector<FeatureType> &features,
      const Fit<PatchworkGPFit<FitModelType, GrouperFunction>> &patchwork_fit,
      PredictTypeIdentity<JointDistribution> &&) const {

    auto get_obs_vector = [](const auto &fit_model) {
      return fit_model.predict(fit_model.get_fit().train_features).mean();
    };

    assert(patchwork_fit.fit_models.size() == 2);

    const auto obs_vectors = patchwork_fit.fit_models.apply(get_obs_vector).values();
    const auto y = concatenate(obs_vectors[0], obs_vectors[1]);

    auto get_features = [](const auto &fit_model) {
      return fit_model.get_fit().train_features;
    };

    auto C_dd_solve = [&](const Eigen::MatrixXd &x) {
      Eigen::Index i = 0;
      Eigen::MatrixXd output(x.rows(), x.cols());
      for (const auto &f : patchwork_fit.fit_models.values()) {
        const auto cov = f.get_fit().train_covariance;
        const auto rhs_chunk = x.block(i, 0, cov.rows(), x.cols());
        output.block(i, 0, cov.rows(), x.cols()) = cov.solve(rhs_chunk);
        i += cov.rows();
      }
      return output;
    };

    auto cov_func_fb = [&](const auto &features, const auto &b_features) {
      Eigen::MatrixXd output = this->covariance_function_(features, b_features);

      for (Eigen::Index i = 0; i < output.rows(); ++i) {
        for (Eigen::Index j = 0; j < output.cols(); ++j) {
          if (!grouper_function_(features[i])) {
            output(i, j) = -output(i, j);
          }
        }
      }

      return output;
    };

    const auto both_features = patchwork_fit.fit_models.apply(get_features).values();
    const auto train_features = concatenate(both_features[0], both_features[1]);

    const auto groups = patchwork_fit.fit_models.keys();
    const auto boundary = boundary_function_(groups[0], groups[1]);

    const Eigen::MatrixXd C_bb = 2. * this->covariance_function_(boundary);
    auto C_db = cov_func_fb(train_features, boundary);

    const Eigen::MatrixXd C_dd_inv_C_db = C_dd_solve(C_db);

    std::cout << C_bb.rows() << " " << C_bb.cols() << std::endl;
    std::cout << C_db.rows() << " " << C_db.cols() << std::endl;
    std::cout << C_dd_inv_C_db.rows() << " " << C_dd_inv_C_db.cols() << std::endl;

    const Eigen::MatrixXd S_bb = C_bb - C_db.transpose() * C_dd_inv_C_db;
    const auto S_bb_ldlt = S_bb.ldlt();
    auto solver = [&](const auto &x) {
      Eigen::MatrixXd output = C_dd_inv_C_db.transpose() * x;
      output = S_bb_ldlt.solve(output);
      output = C_dd_inv_C_db * output;
      output += C_dd_solve(x);
      return output;
    };

    const Eigen::VectorXd new_information = solver(y);

    const Eigen::MatrixXd C_bb_inv_C_bd = C_bb.ldlt().solve(C_db.transpose());

    auto cross_matrix = [&](const auto &features) {
      // Maybe only part of this;
      Eigen::MatrixXd output = this->covariance_function_(features, train_features);

      for (Eigen::Index i = 0; i < output.rows(); ++i) {
        for (Eigen::Index j = 0; j < output.cols(); ++j) {
          if (grouper_function_(features[i]) != grouper_function_(train_features[j])) {
            output(i, j) = 0.;
          }
        }
      }

      output -= cov_func_fb(features, boundary) * C_bb_inv_C_bd;
      return output;
    };


    const auto cross = cross_matrix(features);
    std::cout << cross.rows() << " , " << cross.cols() << std::endl;
    std::cout << C_dd_inv_C_db.rows() << " , " << C_dd_inv_C_db.cols() << std::endl;
    const Eigen::VectorXd pred_mean = cross * new_information;

    Eigen::MatrixXd cov = this->covariance_function_(features);
    const Eigen::MatrixXd C_gb = cov_func_fb(features, boundary);
    cov = cov - C_gb * C_bb.ldlt().solve(C_gb.transpose());
    cov = cov - cross * solver(cross.transpose());

    return JointDistribution(pred_mean, cov);
  }

  template <typename FeatureType>
  auto _fit_impl(const std::vector<FeatureType> &features,
                 const MarginalDistribution &targets) const {

    const auto m = gp_from_covariance(this->covariance_function_, "internal");

    auto create_fit_model = [&](const auto &dataset) {
      return m.fit(dataset);
    };

    const RegressionDataset<FeatureType> dataset(features, targets);

    const auto fit_models = dataset.group_by(grouper_function_).apply(create_fit_model);

    return from_fit_models(fit_models);
  }

  GrouperFunction grouper_function_;
  BoundaryFunction boundary_function_;
};

template <typename CovFunc, typename GrouperFunction,
          typename BoundaryFunction>
inline
PatchworkGaussianProcess<CovFunc, GrouperFunction,
      BoundaryFunction>
patchwork_gp_from_covariance(CovFunc covariance_function,
                               GrouperFunction grouper_function,
                               BoundaryFunction boundary_function) {
  return PatchworkGaussianProcess<CovFunc, GrouperFunction,
      BoundaryFunction>(
      covariance_function, grouper_function, boundary_function);
};

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_MODELS_SPARSE_GP_H_ */
