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


template <typename GroupKey>
inline
Eigen::MatrixXd outer_product(const Grouped<GroupKey, Eigen::MatrixXd> &lhs,
    const Grouped<GroupKey, Eigen::MatrixXd> &rhs) {
  assert(lhs.size() == rhs.size());
  const Eigen::Index rows = lhs.first_value().cols();
  const Eigen::Index cols = rhs.first_value().cols();
  auto has_expected_cols = [](const auto &x) { return x.cols() == cols;};
  assert(rhs.apply(has_expected_cols).all());



}


template <typename GroupKey, typename FeatureType>
struct BoundaryFeature {

  BoundaryFeature(const GroupKey &lhs_, const GroupKey &rhs_, const FeatureType &feature_) :
    lhs(lhs_), rhs(rhs_), feature(feature_) {};

  GroupKey lhs;
  GroupKey rhs;
  FeatureType feature;
};

template <typename CovFuncCaller, typename GroupKey, typename X, typename Y>
inline Eigen::MatrixXd compute_covariance_matrix(CovFuncCaller caller,
                                                 const GroupKey &key,
                                                 const std::vector<X> &xs,
                                                 const std::vector<BoundaryFeature<GroupKey, Y>> &ys) {
  static_assert(is_invocable<CovFuncCaller, X, Y>::value,
                "caller does not support the required arguments");
  static_assert(is_invocable_with_result<CovFuncCaller, double, X, Y>::value,
                "caller does not return a double");
  int m = static_cast<int>(xs.size());
  int n = static_cast<int>(ys.size());
  Eigen::MatrixXd C(m, n);

  int i, j;
  std::size_t si, sj;
  for (i = 0; i < m; i++) {
    si = static_cast<std::size_t>(i);
    for (j = 0; j < n; j++) {
      sj = static_cast<std::size_t>(j);
      if (key == ys[sj].lhs) {
        C(i, j) = caller(xs[si], ys[sj].feature);
      } else if (key == ys[sj].rhs) {
        C(i, j) = -caller(xs[si], ys[sj].feature);
      } else {
        C(i, j) = 0.;
      }
    }
  }
  return C;
}

template <typename CovFuncCaller, typename X, typename GroupKey>
inline Eigen::MatrixXd compute_boundary_covariance_matrix(CovFuncCaller caller,
                                                 const std::vector<BoundaryFeature<GroupKey, X>> &xs,
                                                 const std::vector<BoundaryFeature<GroupKey, X>> &ys) {
  static_assert(is_invocable<CovFuncCaller, X, X>::value,
                "caller does not support the required arguments");
  static_assert(is_invocable_with_result<CovFuncCaller, double, X, X>::value,
                "caller does not return a double");
  int m = static_cast<int>(xs.size());
  int n = static_cast<int>(ys.size());
  Eigen::MatrixXd C(m, n);

  Eigen::Index i, j;
  std::size_t si, sj;
  for (i = 0; i < m; i++) {
    si = static_cast<std::size_t>(i);
    for (j = 0; j < n; j++) {
      sj = static_cast<std::size_t>(j);
      if (xs[si].lhs == ys[sj].lhs && xs[si].rhs == ys[sj].rhs) {
        C(i, j) = 2 * caller(xs[si].feature, ys[sj].feature);
      } else if (xs[si].lhs == ys[sj].lhs && xs[si].rhs != ys[sj].rhs) {
        C(i, j) = caller(xs[si].feature, ys[sj].feature);
      } else if (xs[si].lhs != ys[sj].lhs && xs[si].rhs == ys[sj].rhs) {
        C(i, j) = caller(xs[si].feature, ys[sj].feature);
      } else if (xs[si].lhs == ys[sj].rhs && xs[si].rhs != ys[sj].lhs) {
        C(i, j) = -caller(xs[si].feature, ys[sj].feature);
      } else if (xs[si].lhs != ys[sj].rhs && xs[si].rhs == ys[sj].lhs) {
        C(i, j) = -caller(xs[si].feature, ys[sj].feature);
      } else {
        assert(xs[si].lhs != ys[sj].lhs &&
            xs[si].lhs != ys[sj].rhs &&
            xs[si].rhs != ys[sj].lhs &&
            xs[si].rhs != ys[sj].rhs);
        C(i, j) = 0.;
      }
    }
  }
  return C;
}

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
//  Eigen::MatrixXd C_bb_inv_C_bd;
//  Grouped<GroupKey, Eigen::VectorXd> information;
  GrouperFunction grouper_function;
};

template <typename X, typename Y>
inline
auto block_solve(const X &lhs, const Y &rhs) {
  const auto solve_one_block = [&](const auto &key, const auto &x) {
    const Eigen::MatrixXd output = lhs.at(key).solve(x);
    return output;
  };

  return rhs.apply(solve_one_block);
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

    using GroupKey = typename Fit<PatchworkGPFit<FitModelType, GrouperFunction>>::GroupKey;
    const auto fit_models = patchwork_fit.fit_models;

    auto get_obs_vector = [](const auto &fit_model) {
      return fit_model.predict(fit_model.get_fit().train_features).mean();
    };

    const auto obs_vectors = patchwork_fit.fit_models.apply(get_obs_vector);

    auto get_features = [](const auto &fit_model) {
      return fit_model.get_fit().train_features;
    };

    auto get_train_covariance = [](const auto &fit_model) {
      return fit_model.get_fit().train_covariance;
    };

    const auto C_dd = fit_models.apply(get_train_covariance);

    const auto groups = fit_models.keys();
    using BoundarySubFeatureType = typename decltype(boundary_function_(std::declval<GroupKey>(), std::declval<GroupKey>()))::value_type;
    using BoundaryFeatureType = BoundaryFeature<GroupKey, BoundarySubFeatureType>;
    std::vector<BoundaryFeatureType> boundary_features;
    for (std::size_t i = 0; i < groups.size(); ++i) {
      for (std::size_t j = i + 1; j < groups.size(); ++j) {
        const auto boundary = boundary_function_(groups[i], groups[j]);
        for (const auto &b : boundary) {
          boundary_features.emplace_back(groups[i], groups[j], b);
        }
      }
    }

    const Eigen::MatrixXd C_bb = compute_boundary_covariance_matrix(this->covariance_function_,
        boundary_features, boundary_features);

    auto boundary_cov_func = [&](const auto &key, const auto &fit_model) {
      return compute_covariance_matrix(this->covariance_function_, key, get_features(fit_model),
          boundary_features);
    };

    const auto C_db = fit_models.apply(boundary_cov_func);

    const auto C_dd_inv_C_db = block_solve(C_dd, C_db);
    // After the subsequent steps this'll be:
    //    S_bb = C_bb - C_db * C_dd^-1 * C_db
    Eigen::MatrixXd S_bb = C_bb;
    auto subtract_off_outer_product = [&](const auto &key, const auto &rhs) {
      S_bb -= C_db.at(key).transpose() * rhs;
    };
    C_dd_inv_C_db.apply(subtract_off_outer_product);

    const auto S_bb_ldlt = S_bb.ldlt();

    auto solver = [&](const auto &rhs) {
      // A^-1 rhs + A^-1 C (B - C^T A^-1 C)^-1 C^T A^-1 rhs
      // A^-1 rhs + A^-1 C S^-1 C^T A^-1 rhs

      // A = C_dd
      // B = C_bb
      // C = C_db
      // S_bb = (B - C^T A^-1 C)
      const auto Ai_rhs = block_solve(C_dd, rhs);

      const auto cols = rhs.first_group().second.cols();

      Eigen::MatrixXd SiCtAi_rhs = Eigen::MatrixXd::Zero(C_bb.rows(), cols);
      auto accumulate_SiCtAi = [&](const auto &key, const auto &x) {
        SiCtAi_rhs += S_bb_ldlt.solve(C_db.at(key).transpose() * x);
      };
      Ai_rhs.apply(accumulate_SiCtAi);

      auto product_with_SiCtAi_rhs = [&](const auto &key, const auto &C_db_i) {
        Eigen::MatrixXd output = C_db_i * SiCtAi_rhs;
        return output;
      };
      const auto C_db_SiCtAi_rhs = C_db.apply(product_with_SiCtAi_rhs);

      auto output = block_solve(C_dd, C_db_SiCtAi_rhs);
      // Adds A^-1 rhs to A^-1 C S^-1 C^T A^-1 rhs
      auto add_Ai_rhs = [&](const auto &key, const auto &group) {
        //return group + Ai_rhs.at(key);
        Eigen::MatrixXd output = group + Ai_rhs.at(key);
        return output;
      };
      return output.apply(add_Ai_rhs);
    };

    const auto ys = fit_models.apply(get_obs_vector);
    const auto information = solver(ys);

    Eigen::VectorXd C_bb_inv_C_bd_information = Eigen::VectorXd::Zero(C_bb.rows());
    const auto C_bb_ldlt = C_bb.ldlt();
    auto accumulate_C_bb_inv_C_bd_information = [&](const auto &key, const auto &C_bd_i) {
      C_bb_inv_C_bd_information += C_bb_ldlt.solve(C_bd_i.transpose() * information.at(key));
    };
    C_db.apply(accumulate_C_bb_inv_C_bd_information);

    /*
     * PREDICT
     */

    const auto by_group = group_by(features, grouper_function_);
    const auto indexers = by_group.indexers();

    Eigen::MatrixXd C_fb(features.size(), boundary_features.size());
    for (std::size_t i = 0; i < features.size(); ++i) {
      Eigen::Index ei = static_cast<Eigen::Index>(i);
      std::vector<FeatureType> feature_vector = {features[i]};
      C_fb.row(ei) = compute_covariance_matrix(this->covariance_function_,
          grouper_function_(features[i]), feature_vector, boundary_features);
    }
    const auto C_fb_bb_inv = C_bb_ldlt.solve(C_fb.transpose()).transpose();

    auto compute_cross_block_transpose = [&](const auto &key, const auto &fit_model) {
      const auto train_features = get_features(fit_model);
      Eigen::Index group_size = fit_model.get_fit().train_covariance.rows();

      Eigen::MatrixXd block = Eigen::MatrixXd::Zero(group_size, features.size());
      // Only fill in the rows which correspond to prediction features in
      // the same group.
      for (const auto &idx : indexers.at(key)) {
        std::vector<FeatureType> one_feature = {features[idx]};
        block.col(idx) = this->covariance_function_(one_feature, train_features).row(0);
      }
      block -= C_db.at(key) * C_fb_bb_inv.transpose();
      return block;
    };

    const auto cross_transpose = fit_models.apply(compute_cross_block_transpose);
    const auto C_dd_inv_cross = solver(cross_transpose);

    Eigen::VectorXd mean = Eigen::VectorXd::Zero(features.size());
    Eigen::MatrixXd cov = this->covariance_function_(features, features);
    cov -= C_fb_bb_inv * C_fb.transpose();

    for (const auto &key : by_group.keys()) {
      mean += cross_transpose.at(key).transpose() * information.at(key);
      cov -= cross_transpose.at(key).transpose() * C_dd_inv_cross.at(key);
    };

    return JointDistribution(mean, cov);
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
