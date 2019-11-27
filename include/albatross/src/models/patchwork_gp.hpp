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

/*
 * A BoundaryFeature represents a pseudo observation of the difference
 * between predictions from two different models.  In other words,
 *
 *   BoundaryFeature(key_i, key_j, feature)
 *
 * represents the quantity
 *
 *   model_i.predict(feature) - model_j.predict(feature)
 *
 * Patchwork Krigging uses these to force equivalence between two
 * otherwise independent models.
 */
template <typename GroupKey, typename FeatureType>
struct BoundaryFeature {

  BoundaryFeature(const GroupKey &lhs_, const GroupKey &rhs_, const FeatureType &feature_) :
    lhs(lhs_), rhs(rhs_), feature(feature_) {};

  BoundaryFeature(GroupKey &&lhs_, GroupKey &&rhs_, FeatureType &&feature_) :
    lhs(std::move(lhs_)), rhs(std::move(rhs_)), feature(std::move(feature_)) {};

  GroupKey lhs;
  GroupKey rhs;
  FeatureType feature;
};

template <typename GroupKey, typename FeatureType>
auto as_boundary_feature(GroupKey &&lhs, GroupKey &&rhs,
    FeatureType &&feature) {
  using BoundaryFeatureType = BoundaryFeature<typename std::decay<GroupKey>::type,
                                        typename std::decay<FeatureType>::type>;
  return BoundaryFeatureType(std::forward<GroupKey>(lhs), std::forward<GroupKey>(rhs),
      std::forward<FeatureType>(feature));
}

template <typename GroupKey, typename FeatureType>
auto as_boundary_features(GroupKey &&lhs, GroupKey &&rhs,
                        const std::vector<FeatureType> &features) {
  using BoundaryFeatureType = BoundaryFeature<typename std::decay<GroupKey>::type,
                                        typename std::decay<FeatureType>::type>;

  std::vector<BoundaryFeatureType> boundary_features;
  for (const auto &f : features) {
    boundary_features.emplace_back(as_boundary_feature(lhs, rhs, f));
  }
  return boundary_features;
}

/*
 * GroupFeature
 */

template <typename GroupKey, typename FeatureType>
struct GroupFeature {

  GroupFeature(const GroupKey &key_, const FeatureType &feature_) :
    key(key_), feature(feature_) {};

  GroupFeature(GroupKey &&key_, FeatureType &&feature_) :
    key(std::move(key_)), feature(std::move(feature_)) {};

  GroupKey key;
  FeatureType feature;
};

template <typename GroupKey, typename FeatureType>
auto as_group_feature(GroupKey &&key,
    FeatureType &&feature) {
  using GroupFeatureType = GroupFeature<typename std::decay<GroupKey>::type,
                                        typename std::decay<FeatureType>::type>;
  return GroupFeatureType(std::forward<GroupKey>(key),
      std::forward<FeatureType>(feature));
}

template <typename GrouperFunction, typename FeatureType,
      std::enable_if_t<details::is_valid_grouper<GrouperFunction, FeatureType>::value, int> = 0>
auto as_group_features(const GrouperFunction &group_function,
                       const std::vector<FeatureType> &features) {
  using GroupKey = typename details::grouper_result<GrouperFunction, FeatureType>::type;
  using GroupFeatureType = GroupFeature<GroupKey,
                                        typename std::decay<FeatureType>::type>;

  std::vector<GroupFeatureType> group_features;
  for (const auto &f : features) {
    group_features.emplace_back(as_group_feature(group_function(f), f));
  }
  return group_features;
}

template <typename GroupKey, typename FeatureType,
std::enable_if_t<!details::is_valid_grouper<GroupKey, FeatureType>::value, int> = 0>
auto as_group_features(const GroupKey &key,
                       const std::vector<FeatureType> &features) {
  using GroupFeatureType = GroupFeature<GroupKey,
                                        typename std::decay<FeatureType>::type>;

  std::vector<GroupFeatureType> group_features;
  for (const auto &f : features) {
    group_features.emplace_back(as_group_feature(key, f));
  }
  return group_features;
}

template <typename SubCaller>
struct PatchworkCallerBase {
  template <
      typename CovFunc, typename X, typename Y,
      typename std::enable_if<
          has_valid_cov_caller<CovFunc, SubCaller, X, Y>::value, int>::type = 0>
  static double call(const CovFunc &cov_func, const X &x, const Y &y) {
    return SubCaller::call(cov_func, x, y);
  }

  template <typename CovFunc, typename GroupKey, typename FeatureType>
  static double call(const CovFunc &cov_func,
                     const GroupFeature<GroupKey, FeatureType> &x,
                     const GroupFeature<GroupKey, FeatureType> &y) {
    if (x.key == y.key) {
      return SubCaller::call(cov_func, x.feature, y.feature);
    } else {
      return 0.;
    }
  }

  template <typename CovFunc, typename GroupKey, typename FeatureType>
  static double call(const CovFunc &cov_func,
                     const GroupFeature<GroupKey, FeatureType> &x,
                     const BoundaryFeature<GroupKey, FeatureType> &y) {
    if (x.key == y.lhs) {
      return SubCaller::call(cov_func, x.feature, y.feature);
    } else if (x.key == y.rhs) {
      return -SubCaller::call(cov_func, x.feature, y.feature);
    } else {
      return 0.;
    }
  }

  template <typename CovFunc, typename GroupKey, typename FeatureType>
  static double call(const CovFunc &cov_func,
                     const BoundaryFeature<GroupKey, FeatureType> &x,
                     const BoundaryFeature<GroupKey, FeatureType> &y) {

    if (x.lhs == y.lhs && x.rhs == y.rhs) {
      return 2 * SubCaller::call(cov_func, x.feature, y.feature);
    } else if (x.lhs == y.lhs && x.rhs != y.rhs) {
      return SubCaller::call(cov_func, x.feature, y.feature);
    } else if (x.lhs != y.lhs && x.rhs == y.rhs) {
      return  SubCaller::call(cov_func, x.feature, y.feature);
    } else if (x.lhs == y.rhs && x.rhs != y.lhs) {
      return  -SubCaller::call(cov_func, x.feature, y.feature);
    } else if (x.lhs != y.rhs && x.rhs == y.lhs) {
      return  -SubCaller::call(cov_func, x.feature, y.feature);
    } else {
      return 0.;
    }
  }

};

/*
 * Patchwork GP works by grouping all the features involved which
 * results in several Grouped objects containing block matrix representations.
 *
 * These subsequent methods make those representations easier to work with.
 */

template <typename GroupKey, typename X, typename Y, typename ApplyFunction>
inline Eigen::MatrixXd block_accumulate(const Grouped<GroupKey, X> &lhs,
    const Grouped<GroupKey, Y> &rhs,
    const ApplyFunction &apply_function) {

  static_assert(std::is_same<Eigen::MatrixXd,
      typename invoke_result<ApplyFunction, X, Y>::type>::value,
      "apply_function needs to return an Eigen::MatrixXd type");

  assert(lhs.size() == rhs.size());
  assert(lhs.size() > 0);

  auto one_group = [&](const auto &key) {
    assert(map_contains(lhs, key) && map_contains(rhs, key));
    return apply_function(lhs.at(key), rhs.at(key));
  };

  const auto keys = lhs.keys();
  Eigen::MatrixXd output = one_group(keys[0]);

  for (std::size_t i = 1; i < keys.size(); ++i) {
    output += one_group(keys[i]);
  }

  return output;
}

template <typename GroupKey, typename ApplyFunction>
inline Eigen::MatrixXd block_product(const Grouped<GroupKey, Eigen::MatrixXd> &lhs,
    const Grouped<GroupKey, Eigen::MatrixXd> &rhs) {

  auto matrix_product = [&](const auto &x, const auto &y) {
    return (x * y).eval();
  };

  return block_accumulate(lhs, rhs, matrix_product);
}

template <typename GroupKey>
inline Eigen::MatrixXd block_inner_product(const Grouped<GroupKey, Eigen::MatrixXd> &lhs,
    const Grouped<GroupKey, Eigen::MatrixXd> &rhs) {

  auto matrix_product = [&](const auto &x, const auto &y) {
    return (x.transpose() * y).eval();
  };

  return block_accumulate(lhs, rhs, matrix_product);
}


template <typename GroupKey, typename Solver, typename Rhs>
inline
auto block_solve(const Grouped<GroupKey, Solver> &lhs,
    const Grouped<GroupKey, Rhs> &rhs) {
  const auto solve_one_block = [&](const auto &key, const auto &x) {
    return (lhs.at(key).solve(x)).eval();
  };

  return rhs.apply(solve_one_block);
};

using PatchworkCaller = internal::SymmetricCaller<PatchworkCallerBase<DefaultCaller>>;

template <typename FitModelType, typename PatchworkFunctions>
struct PatchworkGPFit {};

template <typename ModelType, typename FitType, typename PatchworkFunctions>
struct Fit<PatchworkGPFit<FitModel<ModelType, FitType>, PatchworkFunctions>> {

  using FeatureType = typename FitType::Feature;
  using GroupKey = decltype(std::declval<PatchworkFunctions>().grouper(std::declval<FeatureType>()));

  Grouped<GroupKey, FitModel<ModelType, FitType>> fit_models;
//  Eigen::MatrixXd C_bb_inv_C_bd;
//  Grouped<GroupKey, Eigen::VectorXd> information;
  PatchworkFunctions patchwork_functions;

//  using GroupKey = typename details::grouper_result<PatchworkFunctions::GrouperFunction,
//                                                    FeatureType>::type;

  Fit() {};

  template <typename MaybeGroupKey>
  Fit(const Grouped<MaybeGroupKey, FitModel<ModelType, FitType>> &fit_models_,
      PatchworkFunctions patchwork_functions_) :
      fit_models(fit_models_), patchwork_functions(patchwork_functions_) {

    static_assert(std::is_same<MaybeGroupKey, GroupKey>::value,
        "The GroupKey in fit_models_ doesn't match what the grouper function returns");
  };

  Fit(const Grouped<GroupKey, FitModel<ModelType, FitType>> &&fit_models_,
      PatchworkFunctions patchwork_functions_) : fit_models(std::move(fit_models_)),
          patchwork_functions(patchwork_functions_) {};

};

template <typename BoundaryFunction,
          typename GroupKey>
auto build_boundary_features(const BoundaryFunction &boundary_function,
    const std::vector<GroupKey> &groups) {
  using BoundarySubFeatureType = typename invoke_result<BoundaryFunction, GroupKey, GroupKey>::type::value_type;
  using BoundaryFeatureType = BoundaryFeature<GroupKey, BoundarySubFeatureType>;

  std::vector<BoundaryFeatureType> boundary_features;
  for (std::size_t i = 0; i < groups.size(); ++i) {
    for (std::size_t j = i + 1; j < groups.size(); ++j) {
      const auto next_boundary_features = as_boundary_features(groups[i], groups[j],
          boundary_function(groups[i], groups[j]));
      if (next_boundary_features.size() > 0) {
        boundary_features.insert(boundary_features.end(),
            next_boundary_features.begin(),
            next_boundary_features.end());
      }
    }
  }
  assert(boundary_features.size() > 0);
  return boundary_features;
}


template <typename CovFunc, typename PatchworkFunctions>
class PatchworkGaussianProcess
    : public GaussianProcessBase<
          CovFunc, PatchworkGaussianProcess<CovFunc, PatchworkFunctions>> {

public:
  using Base = GaussianProcessBase<
      CovFunc, PatchworkGaussianProcess<CovFunc, PatchworkFunctions>>;

  PatchworkGaussianProcess() : Base() {};
  PatchworkGaussianProcess(CovFunc &covariance_function)
      : Base(covariance_function) {};
  PatchworkGaussianProcess(CovFunc &covariance_function,
      PatchworkFunctions patchwork_functions)
      : Base(covariance_function), patchwork_functions_(patchwork_functions) {
  };

  template <typename FitModelType, typename GroupKey>
  auto from_fit_models(const Grouped<GroupKey, FitModelType> &fit_models) const {
    using PatchworkFitType = Fit<PatchworkGPFit<FitModelType, PatchworkFunctions>>;
    return FitModel<PatchworkGaussianProcess, PatchworkFitType>(*this, PatchworkFitType(fit_models, patchwork_functions_));
  };

  template <typename FeatureType, typename FitModelType>
  JointDistribution _predict_impl(
      const std::vector<FeatureType> &features,
      const Fit<PatchworkGPFit<FitModelType, PatchworkFunctions>> &patchwork_fit,
      PredictTypeIdentity<JointDistribution> &&) const {

    using GroupKey = typename Fit<PatchworkGPFit<FitModelType, PatchworkFunctions>>::GroupKey;

    const auto fit_models = patchwork_fit.fit_models;

    auto get_obs_vector = [](const auto &fit_model) {
      // TOOD: should these be converted to Measurement<> types?
      return fit_model.predict(fit_model.get_fit().train_features).mean();
    };
    const auto obs_vectors = patchwork_fit.fit_models.apply(get_obs_vector);

    auto boundary_function = [&](const GroupKey &x, const GroupKey &y) {
      return patchwork_functions_.boundary(x, y);
    };

    const auto boundary_features = build_boundary_features(boundary_function, fit_models.keys());

    auto patchwork_caller = [&](const auto &x, const auto &y) {
      return PatchworkCaller::call(this->covariance_function_, x, y);
    };

    auto patchwork_covariance_matrix = [&](const auto &xs, const auto &ys) {
      return compute_covariance_matrix(patchwork_caller, xs, ys);
    };

    // C_bb is the covariance matrix between all boundaries, it will
    // have a lot of zeros so could be decomposed more efficiently
    const Eigen::MatrixXd C_bb = patchwork_covariance_matrix(boundary_features, boundary_features);
    const auto C_bb_ldlt = C_bb.ldlt();

    // C_dd is the large block diagonal matrix, with one block for each model
    // or which we already have an efficient way of computing the inverse.
    auto get_train_covariance = [](const auto &fit_model) {
      return fit_model.get_fit().train_covariance;
    };
    const auto C_dd = fit_models.apply(get_train_covariance);

    auto get_features = [](const auto &fit_model) {
      return fit_model.get_fit().train_features;
    };

    // C_db holds the covariance between each model and all boundaries.
    // The actual storage is effectively a map with values which correspond
    // to the covariance between that model's features and the boundaries.
    auto C_db_one_group = [&](const auto &key, const auto &fit_model) {
      const auto group_features = as_group_features(key, get_features(fit_model));
      return patchwork_covariance_matrix(group_features, boundary_features);
    };
    const auto C_db = fit_models.apply(C_db_one_group);
    const auto C_dd_inv_C_db = block_solve(C_dd, C_db);

    //    S_bb = C_bb - C_db * C_dd^-1 * C_db
    const Eigen::MatrixXd S_bb = C_bb - block_inner_product(C_db, C_dd_inv_C_db);
    const auto S_bb_ldlt = S_bb.ldlt();

    auto solver = [&](const auto &rhs) {
      // A^-1 rhs + A^-1 C (B - C^T A^-1 C)^-1 C^T A^-1 rhs
      // A^-1 rhs + A^-1 C S^-1 C^T A^-1 rhs

      // A = C_dd
      // B = C_bb
      // C = C_db
      // S = S_bb = (B - C^T A^-1 C)
      const auto Ai_rhs = block_solve(C_dd, rhs);

      // S_bb^-1 C^T A^-1 rhs
      auto SiCtAi_rhs_block = [&](const Eigen::MatrixXd &C_db_i,
          const auto &Ai_rhs_i) {
        return Eigen::MatrixXd(S_bb_ldlt.solve(C_db_i.transpose() * Ai_rhs_i));
      };
      const Eigen::MatrixXd SiCtAi_rhs = block_accumulate(C_db, Ai_rhs, SiCtAi_rhs_block);

      auto product_with_SiCtAi_rhs = [&](const auto &key, const auto &C_db_i) {
        return Eigen::MatrixXd(C_db_i * SiCtAi_rhs);
      };
      const auto CSiCtAi_rhs = C_db.apply(product_with_SiCtAi_rhs);

      auto output = block_solve(C_dd, CSiCtAi_rhs);
      // Adds A^-1 rhs to A^-1 C S^-1 C^T A^-1 rhs
      auto add_Ai_rhs = [&](const auto &key, const auto &group) {
        return (group + Ai_rhs.at(key)).eval();
      };
      return output.apply(add_Ai_rhs);
    };

    const auto ys = fit_models.apply(get_obs_vector);
    const auto information = solver(ys);

    Eigen::VectorXd C_bb_inv_C_bd_information = Eigen::VectorXd::Zero(C_bb.rows());
    auto accumulate_C_bb_inv_C_bd_information = [&](const auto &key, const auto &C_bd_i) {
      C_bb_inv_C_bd_information += C_bb_ldlt.solve(C_bd_i.transpose() * information.at(key));
    };
    C_db.apply(accumulate_C_bb_inv_C_bd_information);

    /*
     * PREDICT
     */

    auto predict_grouper = [&](const auto &f) {
      return patchwork_functions_.nearest_group(C_db.keys(), patchwork_functions_.grouper(f));
    };

//    const auto by_group = group_by(features, predict_grouper).apply(as_group_features);
//    const auto indexers = by_group.indexers();

    const auto group_features = as_group_features(predict_grouper, features);

    const Eigen::MatrixXd C_fb = patchwork_covariance_matrix(group_features, boundary_features);
    const auto C_fb_bb_inv = C_bb_ldlt.solve(C_fb.transpose()).transpose();

    auto compute_cross_block_transpose = [&](const auto &key, const auto &fit_model) {
      const auto train_features = as_group_features(key, get_features(fit_model));
      Eigen::MatrixXd block = patchwork_covariance_matrix(train_features, group_features);
      block -= C_db.at(key) * C_fb_bb_inv.transpose();
      return block;
    };

    const auto cross_transpose = fit_models.apply(compute_cross_block_transpose);
    const auto C_dd_inv_cross = solver(cross_transpose);

    const Eigen::VectorXd mean = block_inner_product(cross_transpose, information);
    const Eigen::MatrixXd explained = block_inner_product(cross_transpose, C_dd_inv_cross);
    const Eigen::MatrixXd cov = this->covariance_function_(features, features) - C_fb_bb_inv * C_fb.transpose() - explained;

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

    auto grouper = [&](const auto &f) {
      return patchwork_functions_.grouper(f);
    };

    const auto fit_models = dataset.group_by(grouper).apply(create_fit_model);

    return from_fit_models(fit_models);
  }

  PatchworkFunctions patchwork_functions_;
};

template <typename CovFunc, typename PatchworkFunctions>
inline
PatchworkGaussianProcess<CovFunc, PatchworkFunctions>
patchwork_gp_from_covariance(CovFunc covariance_function,
    PatchworkFunctions patchwork_functions) {
  return PatchworkGaussianProcess<CovFunc, PatchworkFunctions>(
      covariance_function, patchwork_functions);
};

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_MODELS_SPARSE_GP_H_ */
