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

#ifndef ALBATROSS_CORE_DECLARATIONS_H
#define ALBATROSS_CORE_DECLARATIONS_H

#include <functional>
#include <map>
#include <vector>

namespace albatross {
template <typename FeatureType> class RegressionModel;
template <typename FeatureType> class RegressionDataset;
template <typename FeatureType, typename FitType>
class SerializableRegressionModel;
template <typename ModelType, typename FeatureType> class GenericRansac;

using FoldIndices = std::vector<std::size_t>;
using FoldName = std::string;
using FoldIndexer = std::map<FoldName, FoldIndices>;
template <typename FeatureType>
using IndexerFunction =
    std::function<FoldIndexer(const RegressionDataset<FeatureType> &)>;

template <typename FeatureType, typename ModelType>
std::unique_ptr<GenericRansac<ModelType, FeatureType>>
make_generic_ransac_model(ModelType *model, double inlier_threshold,
                          std::size_t min_inliers,
                          std::size_t random_sample_size,
                          std::size_t max_iterations,
                          const IndexerFunction<FeatureType> &indexer_function);
}

#endif
