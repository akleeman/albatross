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

#ifndef ALBATROSS_CORE_DISTRIBUTION_H
#define ALBATROSS_CORE_DISTRIBUTION_H

namespace albatross {

/*
 * A Distribution holds what is typically assumed to be a
 * multivariate Gaussian distribution with mean and optional
 * covariance.
 */
template <typename CovarianceType> struct Distribution {
  Eigen::VectorXd mean;
  CovarianceType covariance;

  Distribution() : mean(), covariance(){};
  Distribution(const Eigen::VectorXd &mean_) : mean(mean_), covariance(){};
  Distribution(const Eigen::VectorXd &mean_, const CovarianceType &covariance_)
      : mean(mean_), covariance(covariance_){};

  std::size_t size() const;

  void assert_valid() const;

  bool has_covariance() const;

  double get_diagonal(Eigen::Index i) const;

  bool operator==(const Distribution<CovarianceType> &other) const {
    return (mean == other.mean && covariance == other.covariance);
  }

  template <typename OtherCovarianceType>
  typename std::enable_if<
      !std::is_same<CovarianceType, OtherCovarianceType>::value, bool>::type
  operator==(const Distribution<OtherCovarianceType> &other) const {
    return false;
  }

  /*
   * If the CovarianceType is serializable, add a serialize method.
   */
  template <class Archive>
  typename std::enable_if<
      valid_in_out_serializer<CovarianceType, Archive>::value, void>::type
  serialize(Archive &archive) {
    archive(cereal::make_nvp("mean", mean));
    archive(cereal::make_nvp("covariance", covariance));
  }

  /*
   * If you try to serialize a Distribution for which the covariance
   * type is not serializable you'll get an error.
   */
  template <class Archive>
  typename std::enable_if<
      !valid_in_out_serializer<CovarianceType, Archive>::value, void>::type
  save(Archive &) {
    static_assert(delay_static_assert<Archive>::value,
                  "In order to serialize a Distribution the corresponding "
                  "CovarianceType must be serializable.");
  }
};

template <typename CovarianceType>
std::size_t Distribution<CovarianceType>::size() const {
  // If the covariance is defined it must have the same number
  // of rows and columns which should be the same size as the mean.
  assert_valid();
  return static_cast<std::size_t>(mean.size());
}

template <typename CovarianceType>
void Distribution<CovarianceType>::assert_valid() const {
  if (covariance.size() > 0) {
    assert(covariance.rows() == covariance.cols());
    assert(mean.size() == covariance.rows());
  }
}

template <typename CovarianceType>
bool Distribution<CovarianceType>::has_covariance() const {
  assert_valid();
  return covariance.size() > 0;
}

template <typename CovarianceType>
double Distribution<CovarianceType>::get_diagonal(Eigen::Index i) const {
  return has_covariance() ? covariance.diagonal()[i] : NAN;
}

template <typename SizeType, typename CovarianceType>
Distribution<CovarianceType> subset(const std::vector<SizeType> &indices,
                                    const Distribution<CovarianceType> &dist) {
  auto subset_mean = albatross::subset(indices, Eigen::VectorXd(dist.mean));
  if (dist.has_covariance()) {
    auto subset_cov = albatross::symmetric_subset(indices, dist.covariance);
    return Distribution<CovarianceType>(subset_mean, subset_cov);
  } else {
    return Distribution<CovarianceType>(subset_mean);
  }
}

template <typename SizeType, typename CovarianceType>
void set_subset(const std::vector<SizeType> &indices,
                const Distribution<CovarianceType> &from,
                Distribution<CovarianceType> *to) {
  set_subset(indices, from.mean, &to->mean);
  assert(from.has_covariance() == to->has_covariance());
  if (from.has_covariance()) {
    set_subset(indices, from.covariance, &to->covariance);
  }
}

} // namespace albatross

#endif
