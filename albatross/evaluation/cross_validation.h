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

#ifndef ALBATROSS_CROSSVALIDATION_H
#define ALBATROSS_CROSSVALIDATION_H

namespace albatross {

template <typename ModelType, typename FeatureType>
auto get_predictions(const ModelType &model,
                     const RegressionFolds<FeatureType> &folds) {

  using FitType = typename fit_type<ModelType, FeatureType>::type;
  std::vector<Prediction<ModelType, FeatureType, FitType>> predictions;
  for (const auto &fold : folds) {
    predictions.emplace_back(model.get_fit_model(fold.train_dataset)
                                 .get_prediction(fold.test_dataset.features));
  }

  return predictions;
}

///*
// * Returns a single cross validated prediction distribution
// * for some cross validation folds, taking into account the
// * fact that each fold may contain reordered data.
// */
// template <typename PredictType>
// static inline MarginalDistribution concatenate_fold_predictions(
//    const FoldIndexer &fold_indexer,
//    const std::map<FoldName, PredictType> &predictions) {
//  // Create a new prediction mean that will eventually contain
//  // the ordered concatenation of each fold's predictions.
//  Eigen::Index n = 0;
//  for (const auto &pair : predictions) {
//    n += static_cast<decltype(n)>(pair.second.size());
//  }
//
//  Eigen::VectorXd mean(n);
//  Eigen::VectorXd diagonal(n);
//
//  Eigen::Index number_filled = 0;
//  // Put all the predicted means back in order.
//  for (const auto &pair : predictions) {
//    const auto pred = pair.second;
//    const auto fold_indices = fold_indexer.at(pair.first);
//    assert(pred.size() == fold_indices.size());
//    for (Eigen::Index i = 0; i < pred.mean.size(); i++) {
//      // The test indices map each element in the current fold back
//      // to the original order of the parent dataset.
//      auto test_ind = static_cast<Eigen::Index>(fold_indices[i]);
//      assert(test_ind < n);
//      mean[test_ind] = pred.mean[i];
//      diagonal[test_ind] = pred.get_diagonal(i);
//      number_filled++;
//    }
//  }
//  assert(number_filled == n);
//  return MarginalDistribution(mean, diagonal.asDiagonal());
//}

template <typename ModelType, typename FeatureType>
class Prediction<CrossValidation<ModelType>, FeatureType, FoldIndexer> {
public:
  Prediction(const ModelType &model,
             const RegressionDataset<FeatureType> &dataset,
             const FoldIndexer &indexer)
      : model_(model), dataset_(dataset), indexer_(indexer) {}

  Eigen::VectorXd mean() const {
    const auto folds = folds_from_fold_indexer(dataset_, indexer_);
    const auto predictions = albatross::get_predictions(model_, folds);

    std::vector<Eigen::VectorXd> means;
    for (const auto &pred : predictions) {
      means.emplace_back(pred.mean());
    }

    assert(folds.size() == predictions.size());
    Eigen::Index n = dataset_.size();
    Eigen::VectorXd pred(n);
    Eigen::Index number_filled = 0;
    // Put all the predicted means back in order.
    for (std::size_t i = 0; i < folds.size(); ++i) {
      const auto fold_pred = means[i];
      const auto fold = folds[i];
      Eigen::Index fold_size =
          static_cast<Eigen::Index>(fold.test_dataset.size());
      assert(fold_pred.size() == fold_size);
      set_subset(fold.test_indices, fold_pred, &pred);
      number_filled += static_cast<Eigen::Index>(fold.test_indices.size());
    }
    assert(number_filled == n);
    return pred;
  }

  template <typename DummyType = ModelType,
            typename std::enable_if<
                has_marginal<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value,
                int>::type = 0>
  MarginalDistribution marginal() const {
    const auto folds = folds_from_fold_indexer(dataset_, indexer_);
    const auto predictions = albatross::get_predictions(model_, folds);

    std::vector<MarginalDistribution> marginals;
    for (const auto &pred : predictions) {
      marginals.emplace_back(pred.marginal());
    }

    assert(folds.size() == predictions.size());
    Eigen::Index n = dataset_.size();
    Eigen::VectorXd mean(n);
    Eigen::VectorXd variance(n);
    Eigen::Index number_filled = 0;
    // Put all the predicted means back in order.
    for (std::size_t i = 0; i < folds.size(); ++i) {
      const auto fold_pred = marginals[i];
      const auto fold = folds[i];
      assert(fold_pred.size() == fold.test_dataset.size());
      set_subset(fold.test_indices, fold_pred.mean, &mean);
      set_subset(fold.test_indices, fold_pred.covariance.diagonal(), &variance);
      number_filled += static_cast<Eigen::Index>(fold.test_indices.size());
    }
    assert(number_filled == n);
    return MarginalDistribution(mean, variance.asDiagonal());
  }

  template <typename DummyType = ModelType,
            typename std::enable_if<
                !has_marginal<Prediction<
                    DummyType, FeatureType,
                    typename fit_type<DummyType, FeatureType>::type>>::value,
                int>::type = 0>
  MarginalDistribution marginal() const = delete;
  // marginal is only available if the underlaying model supports it.

  JointDistribution joint() const =
      delete; // Cross validation can't produce a joint distribution.

private:
  const ModelType model_;
  const RegressionDataset<FeatureType> dataset_;
  const FoldIndexer indexer_;
};

template <typename ModelType, typename FeatureType>
using CVPrediction =
    Prediction<CrossValidation<ModelType>, FeatureType, FoldIndexer>;

template <typename ModelType> class CrossValidation {

  ModelType model_;

public:
  CrossValidation(const ModelType &model) : model_(model){};

  template <typename FeatureType>
  auto get_predictions(const RegressionFolds<FeatureType> &folds) const {
    return albatross::get_predictions(model_, folds);
  }

  template <typename FeatureType>
  auto get_predictions(const RegressionDataset<FeatureType> &dataset) const {
    const auto indexer = leave_one_out_indexer(dataset.features);
    const auto folds = folds_from_fold_indexer(dataset, indexer);
    return albatross::get_predictions(model_, folds);
  }

  /*
   * get_prediction deals with returning a Prediction object which looks
   * like any other but which provides out of sample predictions for
   * a given indexer.
   */
  template <typename FeatureType>
  CVPrediction<ModelType, FeatureType>
  get_prediction(const RegressionDataset<FeatureType> &dataset,
                 const FoldIndexer &indexer) const {
    return CVPrediction<ModelType, FeatureType>(model_, dataset, indexer);
  }

  template <typename FeatureType>
  auto get_prediction(const RegressionDataset<FeatureType> &dataset,
                      const GrouperFunction<FeatureType> &grouper) const {
    const auto indexer = leave_one_group_out_indexer(dataset.features, grouper);
    return get_prediction(dataset, indexer);
  }

  template <typename FeatureType>
  auto get_prediction(const RegressionDataset<FeatureType> &dataset) const {
    const auto indexer = leave_one_out_indexer(dataset.features);
    return get_prediction(dataset, indexer);
  }
};

template <typename ModelType>
CrossValidation<ModelType> ModelBase<ModelType>::cross_validate() const {
  return CrossValidation<ModelType>(derived());
}

//
/////*
//// * An evaluation metric is a function that takes a prediction distribution
/// and
//// * corresponding targets and returns a single real value that summarizes
//// * the quality of the prediction.
//// */
//// template <typename PredictType>
//// using EvaluationMetric = std::function<double(
////    const PredictType &prediction, const MarginalDistribution &targets)>;
////
/////*
//// * Iterates over previously computed predictions for each fold and
//// * returns a vector of scores for each fold.
//// */
//// template <typename FeatureType, typename PredictType = JointDistribution>
//// static inline Eigen::VectorXd
//// compute_scores(const EvaluationMetric<PredictType> &metric,
////               const std::vector<RegressionFold<FeatureType>> &folds,
////               const std::vector<PredictType> &predictions) {
////  // Create a vector of metrics, one for each fold.
////  Eigen::VectorXd metrics(static_cast<Eigen::Index>(folds.size()));
////  // Loop over each fold, making predictions then evaluating them
////  // to create the final output.
////  for (Eigen::Index i = 0; i < metrics.size(); i++) {
////    metrics[i] = metric(predictions[i], folds[i].test_dataset.targets);
////  }
////  return metrics;
////}
////
//// template <typename FeatureType, typename CovarianceType>
//// static inline Eigen::VectorXd
//// compute_scores(const EvaluationMetric<Eigen::VectorXd> &metric,
////               const std::vector<RegressionFold<FeatureType>> &folds,
////               const std::vector<Distribution<CovarianceType>> &predictions)
///{
////  std::vector<Eigen::VectorXd> converted;
////  for (const auto &pred : predictions) {
////    converted.push_back(pred.mean);
////  }
////  return compute_scores(metric, folds, converted);
////}
////
/////*
//// * Iterates over each fold in a cross validation set and fits/predicts and
//// * scores the fold, returning a vector of scores for each fold.
//// */
//// template <typename FeatureType, typename PredictType = JointDistribution>
//// static inline Eigen::VectorXd
//// cross_validated_scores(const EvaluationMetric<PredictType> &metric,
////                       const std::vector<RegressionFold<FeatureType>>
///&folds,
////                       RegressionModel<FeatureType> *model) {
////  // Create a vector of predictions.
////  std::vector<PredictType> predictions =
////      model->template cross_validated_predictions<PredictType>(folds);
////  return compute_scores<FeatureType, PredictType>(metric, folds,
/// predictions);
////}
////
/////*
//// * Iterates over each fold in a cross validation set and fits/predicts and
//// * scores the fold, returning a vector of scores for each fold.
//// */
//// template <typename FeatureType, typename PredictType = JointDistribution>
//// static inline Eigen::VectorXd
//// cross_validated_scores(const EvaluationMetric<PredictType> &metric,
////                       const RegressionDataset<FeatureType> &dataset,
////                       const FoldIndexer &fold_indexer,
////                       RegressionModel<FeatureType> *model) {
////  // Create a vector of predictions.
////  std::vector<PredictType> predictions =
////      model->template cross_validated_predictions<PredictType>(dataset,
//// fold_indexer);
////  const auto folds = folds_from_fold_indexer(dataset, fold_indexer);
////  return compute_scores<FeatureType, PredictType>(metric, folds,
/// predictions);
////}
////
////
/////*
//// * Returns a single cross validated prediction distribution
//// * for some cross validation folds, taking into account the
//// * fact that each fold may contain reordered data.
//// */
//// template <typename FeatureType, typename PredictType>
//// static inline MarginalDistribution concatenate_fold_predictions(
////    const std::vector<RegressionFold<FeatureType>> &folds,
////    const std::vector<PredictType> &predictions) {
////
////  // Convert to map variants of the inputs.
////  FoldIndexer fold_indexer;
////  std::map<FoldName, PredictType> prediction_map;
////  for (std::size_t j = 0; j < predictions.size(); j++) {
////    prediction_map[folds[j].name] = predictions[j];
////    fold_indexer[folds[j].name] = folds[j].test_indices;
////  }
////
////  return concatenate_fold_predictions(fold_indexer, prediction_map);
////}
////
//// template <typename FeatureType>
//// static inline MarginalDistribution
//// cross_validated_predict(const std::vector<RegressionFold<FeatureType>>
//// &folds,
////                        RegressionModel<FeatureType> *model) {
////  // Get the cross validated predictions, note however that
////  // depending on the type of folds, these predictions may
////  // be shuffled.
////  const std::vector<MarginalDistribution> predictions =
////      model->template
////      cross_validated_predictions<MarginalDistribution>(folds);
////  return concatenate_fold_predictions(folds, predictions);
////}

} // namespace albatross

#endif
