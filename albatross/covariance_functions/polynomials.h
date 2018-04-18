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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_POLYNOMIALS_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_POLYNOMIALS_H

#include "covariance_term.h"

namespace albatross {

struct ConstantTerm {};

/*
 * The Constant covariance term represents a single scalar
 * value that is shared across all FeatureTypes.  This can
 * be thought of as a mean term.  In fact, the only reason
 * it isn't called a mean term is to avoid ambiguity with the
 * mean of a Gaussian process and because with a prior, the
 * underlying constant term would actually be a biased estimate
 * of the mean.
 */
class Constant : public CovarianceTerm {
 public:
  Constant(double sigma_constant = 10.) {
    this->params_["sigma_constant"] = sigma_constant;
  };

  ~Constant(){};

  std::string get_name() const { return "constant"; }

  template <typename X>
  std::vector<ConstantTerm> get_state_space_representation(std::vector<X> &x) const {
    std::vector<ConstantTerm> terms = {ConstantTerm()};
    return terms;
  }

  /*
   * This will create a covariance matrix that looks like,
   *     sigma_mean^2 * ones(m, n)
   * which is saying all observations are perfectly correlated,
   * so you can move one if you move the rest the same amount.
   */
  template <typename X, typename Y>
  double operator()(const X &x __attribute__((unused)),
                    const Y &y __attribute__((unused))) const {
    double sigma_constant = this->params_.at("sigma_constant");
    return sigma_constant * sigma_constant;
  }
};

}

#endif
