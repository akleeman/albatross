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

#ifndef INCLUDE_ALBATROSS_MODELS_SPARSE_GP_H_
#define INCLUDE_ALBATROSS_MODELS_SPARSE_GP_H_

namespace albatross {

template <typename CovFunc, typename InducingPointStrategy,
          typename IndexingFunction>
class SparseGaussianProcessRegression;

std::vector<double> inline linspace(double a, double b, std::size_t n) {
  double h = (b - a) / static_cast<double>(n - 1);
  std::vector<double> xs(n);
  typename std::vector<double>::iterator x;
  double val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
    *x = val;
  return xs;
}

struct UniformlySpacedInducingPoints {

  UniformlySpacedInducingPoints(std::size_t num_points_ = 10)
      : num_points(num_points_) {}

  std::vector<double> operator()(const std::vector<double> &features) const {
    double min = *std::min_element(features.begin(), features.end());
    double max = *std::max_element(features.begin(), features.end());

    return linspace(min, max, num_points);
  }

  std::size_t num_points;
};

template <typename _Scalar, int _Rows, int _Cols>
inline auto truncated_psd_solve(const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>> &lhs_evd,
                                const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs,
                                double threshold = 1e-8) {
  const auto V = lhs_evd.eigenvectors();
  auto d = lhs_evd.eigenvalues();

  std::vector<Eigen::Index> inds;
  for (Eigen::Index i = 0; i < d.size(); ++i) {
    if (d[i] >= threshold) {
      inds.push_back(i);
    } else {
      std::cout << "    " << d[i] << std::endl;
    }
  }

  const auto V_sub = subset_cols(V, inds);
  const auto d_sub = subset(d, inds);

  Eigen::Matrix<_Scalar, _Rows, _Cols> output = V_sub * d_sub.asDiagonal().inverse() * V_sub.transpose() * rhs;
  return output;
}

/*
 *  This class implements an approximation technique for Gaussian processes
 * which relies on an assumption that all observations are independent (or
 * groups of observations are independent) conditional on a set of inducing
 * points.  The method is based off:
 *
 *     [1] Sparse Gaussian Processes using Pseudo-inputs
 *     Edward Snelson, Zoubin Ghahramani
 *     http://www.gatsby.ucl.ac.uk/~snelson/SPGP_up.pdf
 *
 *  Though the code uses notation closer to that used in this (excellent)
 * overview of these methods:
 *
 *     [2] A Unifying View of Sparse Approximate Gaussian Process Regression
 *     Joaquin Quinonero-Candela, Carl Edward Rasmussen
 *     http://www.jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf
 *
 *  Very broadly speaking this method starts with a prior over the observations,
 *
 *     [f] ~ N(0, K_ff)
 *
 *  where K_ff(i, j) = covariance_function(features[i], features[j]) and f
 * represents the function value.
 *
 *  It then uses a set of inducing points, u, and makes some assumptions about
 * the conditional distribution:
 *
 *     [f|u] ~ N(K_fu K_uu^-1 u, K_ff - Q_ff)
 *
 *  Where Q_ff = K_fu K_uu^-1 K_uf represents the variance in f that is
 * explained by u.
 *
 *  For FITC (Fully Independent Training Contitional) the assumption is that
 * K_ff - Qff is diagonal, for PITC (Partially Independent Training Conditional)
 * that it is block diagonal.  These assumptions lead to an efficient way of
 * inferring the posterior distribution for some new location f*,
 *
 *     [f*|f=y] ~ N(K_*u S K_uf A^-1 y, K_** - Q_** + K_*u S K_u*)
 *
 *  Where S = (K_uu + K_uf A^-1 K_fu)^-1 and A = diag(K_ff - Q_ff) and "diag"
 * may mean diagonal or block diagonal.  Regardless we end up with O(m^2n)
 * complexity instead of O(n^3) of direct Gaussian processes.  (Note that in [2]
 * S is called sigma and A is lambda.)
 *
 *  Of course, the implementation details end up somewhat more complex in order
 * to improve numerical stability.  A few great resources were heavily used to
 * get those deails straight:
 *
 *     - https://bwengals.github.io/pymc3-fitcvfe-implementation-notes.html
 *     - https://github.com/SheffieldML/GPy see fitc.py
 */
template <typename CovFunc, typename InducingPointStrategy,
          typename IndexingFunction>
class SparseGaussianProcessRegression
    : public GaussianProcessBase<
          CovFunc, SparseGaussianProcessRegression<
                       CovFunc, InducingPointStrategy, IndexingFunction>> {

public:
  using Base = GaussianProcessBase<
      CovFunc, SparseGaussianProcessRegression<CovFunc, InducingPointStrategy,
                                               IndexingFunction>>;

  SparseGaussianProcessRegression() : Base(){};
  SparseGaussianProcessRegression(CovFunc &covariance_function)
      : Base(covariance_function){};
  SparseGaussianProcessRegression(
      CovFunc &covariance_function,
      InducingPointStrategy &inducing_point_strategy_,
      IndexingFunction &independent_group_indexing_function_,
      const std::string &model_name)
      : Base(covariance_function, model_name),
        inducing_point_strategy(inducing_point_strategy_),
        independent_group_indexing_function(
            independent_group_indexing_function_){};
  SparseGaussianProcessRegression(CovFunc &covariance_function,
                                  const std::string &model_name)
      : Base(covariance_function, model_name){};

  template <typename FeatureType>
  auto _fit_impl(const std::vector<FeatureType> &out_of_order_features,
                 const MarginalDistribution &out_of_order_targets) const {

    const auto indexer =
        independent_group_indexing_function(out_of_order_features);

    std::vector<std::size_t> reordered_inds;
    BlockDiagonal K_ff;
    for (const auto &pair : indexer) {
      reordered_inds.insert(reordered_inds.end(), pair.second.begin(),
                            pair.second.end());
      auto subset_features = subset(out_of_order_features, pair.second);
      K_ff.blocks.emplace_back(this->covariance_function_(subset_features));
      if (out_of_order_targets.has_covariance()) {
        K_ff.blocks.back().diagonal() +=
            subset(out_of_order_targets.covariance.diagonal(), pair.second);
      }
    }

    const auto features = subset(out_of_order_features, reordered_inds);
    const auto targets = subset(out_of_order_targets, reordered_inds);

    // Determine the set of inducing points, u.
    const auto u = inducing_point_strategy(features);

    Eigen::Index m = static_cast<Eigen::Index>(u.size());

    const Eigen::MatrixXd K_fu = this->covariance_function_(features, u);
    const Eigen::MatrixXd K_uu = this->covariance_function_(u);

    const auto K_uu_llt = K_uu.llt();
    // P is such that:
    //     Q_ff = K_fu K_uu^-1 K_uf
    //          = K_fu L^-T L^-1 K_uf
    //          = P^T P
    const Eigen::MatrixXd P = K_uu_llt.matrixL().solve(K_fu.transpose());

    BlockDiagonal Q_ff;
    Eigen::Index i = 0;
    for (const auto &pair : indexer) {
      Eigen::Index cols = static_cast<Eigen::Index>(pair.second.size());
      auto P_cols = P.block(0, i, P.rows(), cols);
      Q_ff.blocks.emplace_back(P_cols.transpose() * P_cols);
      i += cols;
    }

    auto A = K_ff - Q_ff;

    if (A.diagonal().minCoeff() < 1e-6) {
      // It's possible that the inducing points will perfectly describe
      // some of the data, in which case we need to add a bit of extra
      // noise to make sure lambda is invertible.
      for (auto &b : A.blocks) {
        b.diagonal() += 1e-6 * Eigen::VectorXd::Ones(b.rows());
      }
    }
//    /*
//     *
//     * The end goal here is to produce a vector, v, and matrix, C, such that
//     * for a prediction, f*, we can do,
//     *
//     *     [f*|f=y] ~ N(K_*u * v , K_** - K_*u * C^-1 * K_u*)
//     *
//     *  and it would match the desired prediction described above,
//     *
//     *     [f*|f=y] ~ N(K_*u S K_uf^-1 A^-1 y, K_** − Q_** + K_*u S K_u*)
//     *
//     *  we can find v easily,
//     *
//     *     v = S K_uf A^-1 y
//     *
//     *  and to get C we need to do some algebra,
//     *
//     *     K_** - K_*u * C^-1 * K_u* = K_** - Q_** + K_*u S K_u*
//     *                               = K_** - K_*u (K_uu^-1 - S) K_u*
//     *  which leads to:
//     *     C^-1 = K_uu^-1 - S
//     *                                                  (Expansion of S)
//     *          = K_uu^-1 - (K_uu + K_uf A^-1 K_fu)^-1
//     *                                        (Woodbury Matrix Identity)
//     *          = (K_uu^-1 K_uf (A + K_fu K_uu^-1 K_uf)^-1 K_fu K_uu^-1)
//     *                                   (LL^T = K_uu and P = L^-1 K_uf)
//     *          = L^-T P (A + P^T P)^-1 P^T L^-1
//     *                                        (Searle Set of Identities)
//     *          = L^-T P A^-1 P^T (I + P A^-1 P^T)^-1 L^-1
//     *                         (B = (I + P A^-1 P^T) and R = A^-1/2 P^T)
//     *          = L^-T R^T R B^-1 L^-1
//     *
//     *  taking the inverse of that then gives us:
//     *
//     *      C   = L B (R^T R)^-1 L^T
//     *
//     *  reusing some of the precomputed values there leads to:
//     *
//     *     v = L^-T B^-1 P * A^-1 y
//     */


    const auto A_llt = A.llt();
    const auto A_sqrt = A_llt.matrixL();

    Eigen::Index n = static_cast<Eigen::Index>(features.size());

    Eigen::VectorXd b = Eigen::VectorXd::Zero(n + m);
    b.topRows(n) = A_sqrt.llt().solve(targets.mean);

    Eigen::MatrixXd H = Eigen::MatrixXd(n + m, m);
    H.topRows(n) = A_sqrt.llt().solve(K_fu);
    H.bottomRows(m) = K_uu_llt.matrixL().transpose();

    const auto QR = H.colPivHouseholderQr();
    Eigen::VectorXd v = QR.solve(b);

    Eigen::Index rank = QR.rank();
    const auto R = QR.matrixR().topLeftCorner(rank, rank).template triangularView<Eigen::Upper>();
    const Eigen::MatrixXd Rmat = R;
    const Eigen::MatrixXd Rinv = Rmat.inverse();

    // CHANGE THIS!
    const Eigen::MatrixXd sigma = Rinv * Rinv.transpose();
    const auto C = (K_uu.inverse() - sigma).inverse();

    Eigen::LLT<Eigen::MatrixXd> LLT(K_uu_llt);
    std::cout << "========before=======" << std::endl;
    Eigen::MatrixXd L = LLT.matrixL();
    std::cout << L << std::endl;

    for (Eigen::Index i = 0; i < m; ++i) {
      LLT = LLT.rankUpdate(Rinv.col(i), -1.);
      std::cout << "=======" << i << "==========" << std::endl;
      std::cout << Rinv.col(i).transpose() << std::endl;
      std::cout << Eigen::MatrixXd(LLT.matrixL()) << std::endl;
    }
    L = LLT.matrixL();
    std::cout << "========after=======" << std::endl;
    std::cout << L << std::endl;

    std::cout << "K_uu eigenvals: " << std::endl;
    std::cout << K_uu.bdcSvd().singularValues() << std::endl;

    std::cout << "C eigenvals: " << std::endl;
    std::cout << C.bdcSvd().singularValues() << std::endl;

    std::cout << "sigma eigenvals: " << std::endl;
    std::cout << sigma.bdcSvd().singularValues() << std::endl;

    std::cout << "===============C" << std::endl;
    std::cout << C << std::endl;
    std::cout << "===============V" << std::endl;
    std::cout << v << std::endl;

    Eigen::MatrixXd Pt = P.transpose();
    Eigen::MatrixXd RtR = A_sqrt.llt().solve(Pt);
    RtR = RtR.transpose() * RtR;
    const Eigen::MatrixXd B = Eigen::MatrixXd::Identity(m, m) + RtR;
    const auto B_ldlt = B.ldlt();

    const Eigen::MatrixXd LT = K_uu_llt.matrixL().transpose();
    const Eigen::MatrixXd C_alt = K_uu_llt.matrixL() * B * RtR.ldlt().solve(LT);

    std::cout << "===============C_alt" << std::endl;
    std::cout << C_alt << std::endl;

    Eigen::MatrixXd K_uu_L = K_uu_llt.matrixL();
    std::cout << "========K_uu_L=======" << std::endl;
    std::cout << K_uu_L << std::endl;

    const auto LiR = K_uu_L.inverse() * Rmat.transpose();
    std::cout << "C_3" << std::endl;
    Eigen::MatrixXd C_3 = LiR.transpose() * LiR;
    std::cout << "next" << std::endl;
    C_3 = C_3 - Eigen::MatrixXd::Identity(m, m);

    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> evd(C_3);
    C_3 = truncated_psd_solve(evd, Rmat);

    std::cout << Rmat.rows() << ", " << Rmat.cols() << std::endl;
    C_3 = Rmat.transpose() * C_3;

    std::cout << "===============C_3" << std::endl;
    std::cout << C_3 << std::endl;

    using InducingPointFeatureType = typename std::decay<decltype(u[0])>::type;
    return typename Base::template GPFitType<InducingPointFeatureType>(
        u, C_3.ldlt(), v);
  }

  InducingPointStrategy inducing_point_strategy;
  IndexingFunction independent_group_indexing_function;
};

template <typename CovFunc, typename InducingPointStrategy,
          typename IndexingFunction>
auto sparse_gp_from_covariance(CovFunc covariance_function,
                               InducingPointStrategy &strategy,
                               IndexingFunction &index_function,
                               const std::string &model_name) {
  return SparseGaussianProcessRegression<CovFunc, InducingPointStrategy,
                                         IndexingFunction>(
      covariance_function, strategy, index_function, model_name);
};
} // namespace albatross

#endif /* INCLUDE_ALBATROSS_MODELS_SPARSE_GP_H_ */
