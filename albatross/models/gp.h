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

#ifndef ALBATROSS_MODELS_GP_H
#define ALBATROSS_MODELS_GP_H

namespace albatross {

///* Forward Declarations */
// template <typename FeatureType, typename CovarianceType>
// class GaussianProcessRansac;
// template <typename FeatureType, typename CovarianceType>
// class GaussianProcessRegression;
// template <typename FeatureType, typename CovarianceType>
// inline std::unique_ptr<GaussianProcessRansac<FeatureType, CovarianceType>>
// make_gp_ransac_model(
//    GaussianProcessRegression<FeatureType, CovarianceType> *model,
//    double inlier_threshold, std::size_t min_inliers,
//    std::size_t random_sample_size, std::size_t max_iterations,
//    const IndexerFunction<FeatureType> &indexer_function);

template <typename CovFunc, typename ImplType, typename FeatureType>
struct Fit<GaussianProcessBase<CovFunc, ImplType>, FeatureType> {

  std::vector<FeatureType> train_features;
  Eigen::SerializableLDLT train_ldlt;
  Eigen::VectorXd information;

  Fit(){};

  Fit(const std::vector<FeatureType> &features,
      const Eigen::MatrixXd &train_cov, const MarginalDistribution &targets) {
    train_features = features;
    Eigen::MatrixXd cov(train_cov);
    if (targets.has_covariance()) {
      cov += targets.covariance;
    }
    train_ldlt = Eigen::SerializableLDLT(cov.ldlt());
    // Precompute the information vector
    information = train_ldlt.solve(targets.mean);
  }

  template <typename Archive>
  // todo: enable if FeatureType is serializable
  void serialize(Archive &archive) {
    archive(cereal::make_nvp("information", information));
    archive(cereal::make_nvp("train_ldlt", train_ldlt));
    archive(cereal::make_nvp("train_features", train_features));
  }

  bool operator==(const Fit &other) const {
    return (train_features == other.train_features &&
            train_ldlt == other.train_ldlt && information == other.information);
  }
};

inline JointDistribution predict_from_covariance_and_fit(
    const Eigen::MatrixXd &cross_cov, const Eigen::MatrixXd &pred_cov,
    const Eigen::VectorXd &information, const Eigen::SerializableLDLT &ldlt) {
  const Eigen::VectorXd pred = cross_cov.transpose() * information;
  Eigen::MatrixXd posterior_cov = ldlt.solve(cross_cov);
  posterior_cov = cross_cov.transpose() * posterior_cov;
  posterior_cov = pred_cov - posterior_cov;
  return JointDistribution(pred, posterior_cov);
}

template <typename CovFunc, typename ImplType>
class GaussianProcessBase
    : public ModelBase<GaussianProcessBase<CovFunc, ImplType>> {

  template <typename FitFeatureType>
  using GPFitType = Fit<GaussianProcessBase<CovFunc, ImplType>, FitFeatureType>;

public:
  GaussianProcessBase()
      : covariance_function_(), model_name_(covariance_function_.get_name()){};
  GaussianProcessBase(CovFunc &covariance_function)
      : covariance_function_(covariance_function),
        model_name_(covariance_function_.get_name()){};
  /*
   * Sometimes it's nice to be able to provide a custom model name since
   * these models are generalizable.
   */
  GaussianProcessBase(CovFunc &covariance_function,
                      const std::string &model_name)
      : covariance_function_(covariance_function), model_name_(model_name){};
  GaussianProcessBase(const std::string &model_name)
      : covariance_function_(), model_name_(model_name){};

  ~GaussianProcessBase(){};

  std::string get_name() const { return model_name_; };

  //  template <typename Archive> void save(Archive &archive) const {
  //    archive(cereal::base_class<SerializableRegressionModel<
  //                FeatureType, GaussianProcessFit<FeatureType>>>(this));
  //    archive(model_name_);
  //  }
  //
  //  template <typename Archive> void load(Archive &archive) {
  //    archive(cereal::base_class<SerializableRegressionModel<
  //                FeatureType, GaussianProcessFit<FeatureType>>>(this));
  //    archive(model_name_);
  //  }

  /*
   * The Gaussian Process Regression model derives its parameters from
   * the covariance functions.
   */
  ParameterStore get_params() const override {
    return map_join(impl().params_, covariance_function_.get_params());
  }

  void unchecked_set_param(const std::string &name,
                           const Parameter &param) override {

    if (map_contains(covariance_function_.get_params(), name)) {
      covariance_function_.set_param(name, param);
    } else {
      impl().params_[name] = param;
    }
  }

  std::string pretty_string() const {
    std::ostringstream ss;
    ss << "model_name: " << get_name() << std::endl;
    ss << "covariance_name: " << covariance_function_.pretty_string();
    ss << "has_been_fit: " << this->has_been_fit() << std::endl;
    return ss.str();
  }

  // If the implementing class doesn't have a fit method for this
  // FeatureType but the CovarianceFunction does.
  template <typename FeatureType,
            typename std::enable_if<
                has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
                    !has_valid_fit<ImplType, FeatureType>::value,
                int>::type = 0>
  GPFitType<FeatureType> fit(const std::vector<FeatureType> &features,
                             const MarginalDistribution &targets) const {
    Eigen::MatrixXd cov = covariance_function_(features);
    return GPFitType<FeatureType>(features, cov, targets);
  }

  template <typename FeatureType,
            typename std::enable_if<has_valid_fit<ImplType, FeatureType>::value,
                                    int>::type = 0>
  auto fit(const std::vector<FeatureType> &features,
           const MarginalDistribution &targets) const {
    return impl().fit(features, targets);
  }

  template <
      typename FeatureType, typename FitFeaturetype,
      typename std::enable_if<
          has_call_operator<CovFunc, FeatureType, FeatureType>::value &&
              has_call_operator<CovFunc, FeatureType, FitFeaturetype>::value &&
              !has_valid_predict<ImplType, FeatureType,
                                 GPFitType<FitFeaturetype>,
                                 JointDistribution>::value,
          int>::type = 0>
  JointDistribution predict(const std::vector<FeatureType> &features,
                            const GPFitType<FitFeaturetype> &gp_fit,
                            PredictTypeIdentity<JointDistribution> &&) const {
    const auto cross_cov =
        covariance_function_(gp_fit.train_features, features);
    Eigen::MatrixXd pred_cov = covariance_function_(features);
    return predict_from_covariance_and_fit(
        cross_cov, pred_cov, gp_fit.information, gp_fit.train_ldlt);
  }

  template <typename FeatureType, typename FitFeatureType, typename PredictType,
            typename std::enable_if<has_valid_predict<ImplType, FeatureType,
                                                      GPFitType<FitFeatureType>,
                                                      PredictType>::value,
                                    int>::type = 0>
  PredictType predict(const std::vector<FeatureType> &features,
                      const GPFitType<FitFeatureType> &gp_fit,
                      PredictTypeIdentity<PredictType> &&) const {
    return impl().predict(features, gp_fit, PredictTypeIdentity<PredictType>());
  }

  template <
      typename FeatureType, typename FitFeatureType, typename PredictType,
      typename std::enable_if<
          (!has_call_operator<CovFunc, FeatureType, FeatureType>::value ||
           !has_call_operator<CovFunc, FeatureType, FitFeatureType>::value) &&
              !has_valid_predict<ImplType, FeatureType,
                                 GPFitType<FitFeatureType>, PredictType>::value,
          int>::type = 0>
  PredictType predict(const std::vector<FeatureType> &features,
                      const GPFitType<FitFeatureType> &gp_fit,
                      PredictTypeIdentity<PredictType> &&) const =
      delete; // Covariance Function isn't defined for FeatureType.

  //  virtual MarginalDistribution
  //  predict_marginal_(const std::vector<FeatureType> &features) const override
  //  {
  //    const auto cross_cov =
  //        covariance_function_(features, this->model_fit_.train_features);
  //    const Eigen::VectorXd pred = cross_cov * this->model_fit_.information;
  //    // Here we efficiently only compute the diagonal of the posterior
  //    // covariance matrix.
  //    auto ldlt = this->model_fit_.train_ldlt;
  //    Eigen::MatrixXd explained = ldlt.solve(cross_cov.transpose());
  //    Eigen::VectorXd marginal_variance =
  //        -explained.cwiseProduct(cross_cov.transpose()).array().colwise().sum();
  //    for (Eigen::Index i = 0; i < pred.size(); i++) {
  //      marginal_variance[i] += covariance_function_(features[i],
  //      features[i]);
  //    }
  //
  //    return MarginalDistribution(pred, marginal_variance.asDiagonal());
  //  }
  //
  //  virtual Eigen::VectorXd
  //  predict_mean_(const std::vector<FeatureType> &features) const override {
  //    const auto cross_cov =
  //        covariance_function_(features, this->model_fit_.train_features);
  //    const Eigen::VectorXd pred = cross_cov * this->model_fit_.information;
  //    return pred;
  //  }

  //  virtual std::unique_ptr<RegressionModel<FeatureType>>
  //  ransac_model(double inlier_threshold, std::size_t min_inliers,
  //               std::size_t random_sample_size,
  //               std::size_t max_iterations) override {
  //    static_assert(
  //        is_complete<
  //            GaussianProcessRansac<FeatureType, CovFunc>>::value,
  //        "ransac methods aren't complete yet, be sure you've included "
  //        "ransac_gp.h");
  //    return make_gp_ransac_model<FeatureType, CovFunc>(
  //        this, inlier_threshold, min_inliers, random_sample_size,
  //        max_iterations,
  //        leave_one_out_indexer<FeatureType>);
  //  }

protected:
  //  /*
  //   * Cross validation specializations
  //   *
  //   * The leave one out cross validated predictions for a Gaussian Process
  //   * can be efficiently computed by dropping rows and columns from the
  //   * covariance and obtaining the prediction for the dropped index.  This
  //   * results in something like,
  //   *
  //   *     mean[group] = y[group] - A^{-1} (C^{-1} y)[group]
  //   *     variance[group] = A^{-1}
  //   *
  //   * with group the set of indices for the held out group and
  //   *     A = C^{-1}[group, group]
  //   * is the block of the inverse of the covariance that corresponds
  //   * to the group in question.
  //   *
  //   * See section 5.4.2 Rasmussen Gaussian Processes
  //   */
  //  virtual std::vector<JointDistribution> cross_validated_predictions_(
  //      const RegressionDataset<FeatureType> &dataset,
  //      const FoldIndexer &fold_indexer,
  //      const detail::PredictTypeIdentity<JointDistribution> &) override {
  //
  //    this->fit(dataset);
  //    const FitType model_fit = this->get_fit();
  //    const std::vector<FoldIndices> indices = map_values(fold_indexer);
  //    const auto inverse_blocks =
  //    model_fit.train_ldlt.inverse_blocks(indices);
  //
  //    std::vector<JointDistribution> output;
  //    for (std::size_t i = 0; i < inverse_blocks.size(); i++) {
  //      Eigen::VectorXd yi = subset(indices[i], dataset.targets.mean);
  //      Eigen::VectorXd vi = subset(indices[i], model_fit.information);
  //      const auto A_inv = inverse_blocks[i].inverse();
  //      output.push_back(JointDistribution(yi - A_inv * vi, A_inv));
  //    }
  //    return output;
  //  }
  //
  //  virtual std::vector<MarginalDistribution> cross_validated_predictions_(
  //      const RegressionDataset<FeatureType> &dataset,
  //      const FoldIndexer &fold_indexer,
  //      const detail::PredictTypeIdentity<MarginalDistribution> &) override {
  //    this->fit(dataset);
  //    const FitType model_fit = this->get_fit();
  //
  //    const std::vector<FoldIndices> indices = map_values(fold_indexer);
  //    const auto inverse_blocks =
  //    model_fit.train_ldlt.inverse_blocks(indices);
  //
  //    std::vector<MarginalDistribution> output;
  //    for (std::size_t i = 0; i < inverse_blocks.size(); i++) {
  //      Eigen::VectorXd yi = subset(indices[i], dataset.targets.mean);
  //      Eigen::VectorXd vi = subset(indices[i], model_fit.information);
  //      const auto A_ldlt = Eigen::SerializableLDLT(inverse_blocks[i].ldlt());
  //
  //      output.push_back(MarginalDistribution(
  //          yi - A_ldlt.solve(vi), A_ldlt.inverse_diagonal().asDiagonal()));
  //    }
  //    return output;
  //  }
  //
  //  virtual std::vector<Eigen::VectorXd> cross_validated_predictions_(
  //      const RegressionDataset<FeatureType> &dataset,
  //      const FoldIndexer &fold_indexer,
  //      const detail::PredictTypeIdentity<PredictMeanOnly> &) override {
  //    this->fit(dataset);
  //    const FitType model_fit = this->get_fit();
  //
  //    const std::vector<FoldIndices> indices = map_values(fold_indexer);
  //    const auto inverse_blocks =
  //    model_fit.train_ldlt.inverse_blocks(indices);
  //
  //    std::vector<Eigen::VectorXd> output;
  //    for (std::size_t i = 0; i < inverse_blocks.size(); i++) {
  //      Eigen::VectorXd yi = subset(indices[i], dataset.targets.mean);
  //      Eigen::VectorXd vi = subset(indices[i], model_fit.information);
  //      const auto A_ldlt = Eigen::SerializableLDLT(inverse_blocks[i].ldlt());
  //      output.push_back(yi - A_ldlt.solve(vi));
  //    }
  //    return output;
  //  }

  /*
   * CRTP Helpers
   */
  ImplType &impl() { return *static_cast<ImplType *>(this); }
  const ImplType &impl() const { return *static_cast<const ImplType *>(this); }

  CovFunc covariance_function_;
  std::string model_name_;
};

template <typename CovFunc>
class GaussianProcessRegression
    : public GaussianProcessBase<CovFunc, GaussianProcessRegression<CovFunc>> {
public:
  using Base = GaussianProcessBase<CovFunc, GaussianProcessRegression<CovFunc>>;

  GaussianProcessRegression() : Base(){};
  GaussianProcessRegression(CovFunc &covariance_function)
      : Base(covariance_function){};
  GaussianProcessRegression(CovFunc &covariance_function,
                            const std::string &model_name)
      : Base(covariance_function, model_name){};

  // The only reason these are here is to hide the base class implementations.
  void fit() const = delete;
  void predict() const = delete;
};

template <typename CovFunc>
auto gp_from_covariance(CovFunc covariance_function,
                        const std::string &model_name) {
  return GaussianProcessRegression<CovFunc>(covariance_function, model_name);
};

template <typename CovFunc>
auto gp_from_covariance(CovFunc covariance_function) {
  return GaussianProcessRegression<CovFunc>(covariance_function,
                                            covariance_function.get_name());
};

// template <typename FeatureType, typename CovFunc, typename ImplType =
// NullGPImpl>
// GaussianProcessRegression<FeatureType, CovFunc, ImplType>
// gp_from_covariance(CovFunc covariance_function, const std::string &model_name
// = covariance_function.get_name()) {
//  return GaussianProcessRegression<FeatureType, CovFunc,
//  ImplType>(covariance_function,
//                                                         model_name);
//};

} // namespace albatross

#endif
