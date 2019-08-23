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

#ifndef INCLUDE_ALBATROSS_UTILS_BLOCK_UTILS_H_
#define INCLUDE_ALBATROSS_UTILS_BLOCK_UTILS_H_

namespace albatross {

inline
void print_matrix_stats(const std::string &name, const Eigen::MatrixXd &x) {
//  const Eigen::VectorXd evals = x.eigenvalues().real();
//  double min = evals.minCoeff();
//  double max = evals.maxCoeff();
//  double cond = max / min;
//  const Eigen::MatrixXd asymm = x - x.transpose();
//  std::cout << name << "     min: " << min << " max: " << max << " cond: " << cond << " asym norm : " << asymm.norm() << " asym max : " << asymm.array().abs().maxCoeff() << std::endl;
}

template <typename MatrixType, unsigned int Mode = Eigen::Lower>
struct BlockTriangularView;

struct BlockDiagonalLLT;
struct BlockDiagonal;

struct BlockDiagonalLLT {
  std::vector<Eigen::LLT<Eigen::MatrixXd>> blocks;

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  BlockTriangularView<const Eigen::MatrixXd> matrixL() const;

  Eigen::Index rows() const;

  Eigen::Index cols() const;
};

template <typename MatrixType, unsigned int Mode> struct BlockTriangularView {
  std::vector<Eigen::TriangularView<MatrixType, Mode>> blocks;

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  operator*(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  Eigen::Index rows() const;

  Eigen::Index cols() const;

  Eigen::MatrixXd toDense() const;
};

struct BlockDiagonal {
  std::vector<Eigen::MatrixXd> blocks;

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  operator*(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  BlockDiagonal operator-(const BlockDiagonal &rhs) const;

  Eigen::VectorXd diagonal() const;

  BlockDiagonalLLT llt() const;

  Eigen::Index rows() const;

  Eigen::Index cols() const;

  Eigen::MatrixXd toDense() const;
};

template <typename Derived>
inline
Derived sqrt_solve(const DirectInverse &A,
    const Eigen::MatrixBase<Derived> &b) {
  return A.inverse_.llt().matrixL() * b;
}

template <typename Derived>
inline
Derived sqrt_solve_transpose(const DirectInverse &A,
    const Eigen::MatrixBase<Derived> &b) {
  return A.inverse_.llt().matrixL().transpose() * b;
}


template <typename Derived>
inline
Derived sqrt_solve(const Eigen::LDLT<Eigen::MatrixXd, Eigen::Lower> &A,
    const Eigen::MatrixBase<Derived> &b) {

  Derived output = A.transpositionsP() * b;

  // output = L^-1 (P b)
  A.matrixL().solveInPlace(output);

  // dst = D^-1/2 (L^-1 P b)
  using std::abs;
  const auto vecD = A.vectorD();
  double tolerance = 1. / Eigen::NumTraits<double>::highest();

  for (Eigen::Index i = 0; i < vecD.size(); ++i) {
    if(fabs(vecD(i)) > tolerance) {
      output.row(i) /= sqrt(fabs(vecD(i)));
    } else {
//      std::cout << "WARNING: INVALID DIAGONAL : " << vecD(i) << std::endl;
      output.row(i).setZero();
    }
  }

  return output;
}

template <typename Derived>
inline
Derived sqrt_solve_transpose(const Eigen::LDLT<Eigen::MatrixXd, Eigen::Lower> &A,
    const Eigen::MatrixBase<Derived> &b) {

  Derived output = b;

  // output = D^-1/2 b
  using std::abs;
  const auto vecD = A.vectorD();
  double tolerance = 1. / Eigen::NumTraits<double>::highest();
  for (Eigen::Index i = 0; i < vecD.size(); ++i) {
    if(fabs(vecD(i)) > tolerance) {
      output.row(i) /= sqrt(fabs(vecD(i)));
    } else {
//      std::cout << "WARNING: INVALID DIAGONAL : " << vecD(i) << std::endl;
      output.row(i).setZero();
    }
  }

  // output = L^-1 (D^-1/2 b)
  A.matrixL().transpose().solveInPlace(output);

  // output = P^T L^-1 (D^-1/2 b)
  output = A.transpositionsP().transpose() * output;

  return output;
}

inline
Eigen::MatrixXd schur_complement(const Eigen::SerializableLDLT &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &C) {
  const Eigen::MatrixXd A_sqrti_B = sqrt_solve(A, B);
  const Eigen::MatrixXd explained = A_sqrti_B.transpose() * A_sqrti_B;
  const Eigen::MatrixXd output = C - explained;
  return output;
}

template <typename Solver> struct BlockSymmetric {

  /*
   * Stores a covariance matrix which takes the form:
   *
   *   X = |A   B|
   *       |B.T C|
   *
   * It is assumes that both A and C - B.T A^-1 B are invertible and is
   * designed for the situation where A is larger than C.  The primary
   * use case is for a situation where you have a pre computed LDLT of
   * a submatrix (A) and you'd like to perform a solve of the larger
   * matrix (X)
   *
   * To do so the rules for block inversion are used:
   *
   * https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion
   *
   * which leads to:
   *
   *   X^-1 = |A   B|^-1
   *          |B.T C|
   *
   *        = |A^-1 + Ai_B S^-1 Ai_B^T    -Ai_B S^-1|
   *          |S^-1 Ai_B^T                    S^-1  |
   *
   * where Ai_B = A^-1 B  and S = C - B^T A^_1 B.
   *
   * In this particular implementation Ai_B and S^-1 are pre-computed.
   */
  BlockSymmetric(){};

  BlockSymmetric(const Solver &A_, const Eigen::MatrixXd &B_,
                 const Eigen::SerializableLDLT &S_)
      : A(A_), Li_B(albatross::sqrt_solve(A_, B_)), S(S_) {}

  BlockSymmetric(const Solver &A_, const Eigen::MatrixXd &B_,
                 const Eigen::MatrixXd &C)
      : BlockSymmetric(
            A_, B_,
            Eigen::SerializableLDLT(schur_complement(A_, B_, C))){};

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  sqrt_solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  bool operator==(const BlockSymmetric &rhs) const;

  template <typename Archive>
  void serialize(Archive &archive, const std::uint32_t);

  Eigen::Index rows() const;

  Eigen::Index cols() const;

  Solver A;
  Eigen::MatrixXd Li_B;
  Eigen::SerializableLDLT S;
};

/*
 * BlockDiagonalLLT
 */
template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols>
BlockDiagonalLLT::solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
  assert(cols() == rhs.rows());
  Eigen::Index i = 0;
  Eigen::Matrix<_Scalar, _Rows, _Cols> output(rows(), rhs.cols());
  for (const auto &b : blocks) {
    const auto rhs_chunk = rhs.block(i, 0, b.rows(), rhs.cols());
    output.block(i, 0, b.rows(), rhs.cols()) = b.solve(rhs_chunk);
    i += b.rows();
  }
  return output;
}

inline BlockTriangularView<const Eigen::MatrixXd>
BlockDiagonalLLT::matrixL() const {
  BlockTriangularView<const Eigen::MatrixXd> output;
  for (const auto &b : blocks) {
    Eigen::TriangularView<const Eigen::MatrixXd, Eigen::Lower> L = b.matrixL();
    output.blocks.push_back(L);
  }
  return output;
}

inline Eigen::Index BlockDiagonalLLT::rows() const {
  Eigen::Index n = 0;
  for (const auto &b : blocks) {
    n += b.rows();
  }
  return n;
}

inline Eigen::Index BlockDiagonalLLT::cols() const {
  Eigen::Index n = 0;
  for (const auto &b : blocks) {
    n += b.cols();
  }
  return n;
}

/*
 * BlockTriangularView
 */
template <typename MatrixType, unsigned int Mode>
template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols>
BlockTriangularView<MatrixType, Mode>::solve(
    const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
  assert(cols() == rhs.rows());
  Eigen::Index i = 0;
  Eigen::Matrix<_Scalar, _Rows, _Cols> output(rows(), rhs.cols());
  for (const auto &b : blocks) {
    const auto rhs_chunk = rhs.block(i, 0, b.rows(), rhs.cols());
    output.block(i, 0, b.rows(), rhs.cols()) = b.solve(rhs_chunk);
    i += b.rows();
  }
  return output;
}

template <typename MatrixType, unsigned int Mode>
template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols>
    BlockTriangularView<MatrixType, Mode>::
    operator*(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
  assert(cols() == rhs.rows());
  Eigen::Index i = 0;
  Eigen::Matrix<_Scalar, _Rows, _Cols> output =
      Eigen::Matrix<_Scalar, _Rows, _Cols>::Zero(rows(), rhs.cols());
  for (const auto &b : blocks) {
    const auto rhs_chunk = rhs.block(i, 0, b.rows(), rhs.cols());
    output.block(i, 0, b.rows(), rhs.cols()) = b * rhs_chunk;
    i += b.rows();
  }
  return output;
}

template <typename MatrixType, unsigned int Mode>
inline Eigen::Index BlockTriangularView<MatrixType, Mode>::rows() const {
  Eigen::Index n = 0;
  for (const auto &b : blocks) {
    n += b.rows();
  }
  return n;
}

template <typename MatrixType, unsigned int Mode>
inline Eigen::Index BlockTriangularView<MatrixType, Mode>::cols() const {
  Eigen::Index n = 0;
  for (const auto &b : blocks) {
    n += b.cols();
  }
  return n;
}

template <typename MatrixType, unsigned int Mode>
inline Eigen::MatrixXd BlockTriangularView<MatrixType, Mode>::toDense() const {
  Eigen::MatrixXd output = Eigen::MatrixXd::Zero(rows(), cols());

  Eigen::Index i = 0;
  Eigen::Index j = 0;
  for (const auto &b : blocks) {
    output.block(i, j, b.rows(), b.cols()) = b;
    i += b.rows();
    j += b.cols();
  }
  return output;
}

/*
 * Block Diagonal
 */

template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols> BlockDiagonal::
operator*(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
  assert(cols() == rhs.rows());
  Eigen::Index i = 0;
  Eigen::Matrix<_Scalar, _Rows, _Cols> output =
      Eigen::Matrix<_Scalar, _Rows, _Cols>::Zero(rows(), rhs.cols());
  for (const auto &b : blocks) {
    const auto rhs_chunk = rhs.block(i, 0, b.rows(), rhs.cols());
    output.block(i, 0, b.rows(), rhs.cols()) = b * rhs_chunk;
    i += b.rows();
  }
  return output;
}

inline BlockDiagonal BlockDiagonal::operator-(const BlockDiagonal &rhs) const {
  assert(cols() == rhs.rows());
  assert(blocks.size() == rhs.blocks.size());

  BlockDiagonal output;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    assert(blocks[i].size() == rhs.blocks[i].size());
    output.blocks.emplace_back(blocks[i] - rhs.blocks[i]);
  }
  return output;
}

inline Eigen::MatrixXd BlockDiagonal::toDense() const {
  Eigen::MatrixXd output = Eigen::MatrixXd::Zero(rows(), cols());

  Eigen::Index i = 0;
  Eigen::Index j = 0;
  for (const auto &b : blocks) {
    output.block(i, j, b.rows(), b.cols()) = b;
    i += b.rows();
    j += b.cols();
  }
  return output;
}

inline Eigen::VectorXd BlockDiagonal::diagonal() const {
  assert(rows() == cols());
  Eigen::VectorXd output(rows());

  Eigen::Index i = 0;
  for (const auto b : blocks) {
    output.block(i, 0, b.rows(), 1) = b.diagonal();
    i += b.rows();
  }
  return output;
}

inline BlockDiagonalLLT BlockDiagonal::llt() const {
  BlockDiagonalLLT output;
  for (const auto &b : blocks) {
    output.blocks.emplace_back(b.llt());
  }
  return output;
}

inline Eigen::Index BlockDiagonal::rows() const {
  Eigen::Index n = 0;
  for (const auto &b : blocks) {
    n += b.rows();
  }
  return n;
}

inline Eigen::Index BlockDiagonal::cols() const {
  Eigen::Index n = 0;
  for (const auto &b : blocks) {
    n += b.cols();
  }
  return n;
}

/*
 * BlockSymmetric
 *
 */
template <typename Solver>
template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols> BlockSymmetric<Solver>::solve(
    const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
  Eigen::Index n = A.rows() + S.rows();
  assert(rhs.rows() == n);

  const Eigen::MatrixXd rhs_x = rhs.topRows(A.rows());
  Eigen::MatrixXd rhs_y = rhs.bottomRows(S.rows());

  Eigen::MatrixXd x_hat = albatross::sqrt_solve(A, rhs_x);
  rhs_y = rhs_y - Li_B.transpose() * x_hat;
  const Eigen::MatrixXd y_hat = albatross::sqrt_solve(S, rhs_y);

  const auto y = sqrt_solve_transpose(S, y_hat);

  x_hat -= Li_B * y;
  const auto x = sqrt_solve_transpose(A, x_hat);

  Eigen::Matrix<_Scalar, _Rows, _Cols> output(n, rhs.cols());
  output.topRows(A.rows()) = x;
  output.bottomRows(S.rows()) = y;
  return output;
}

template <typename Solver>
template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols> BlockSymmetric<Solver>::sqrt_solve(
    const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
  Eigen::Index n = A.rows() + S.rows();
  assert(rhs.rows() == n);

  const Eigen::MatrixXd rhs_x = rhs.topRows(A.rows());
  Eigen::MatrixXd rhs_y = rhs.bottomRows(S.rows());

  Eigen::MatrixXd x_hat = albatross::sqrt_solve(A, rhs_x);
  rhs_y = rhs_y - Li_B.transpose() * x_hat;
  const Eigen::MatrixXd y_hat = albatross::sqrt_solve(S, rhs_y);

  Eigen::Matrix<_Scalar, _Rows, _Cols> output(n, rhs.cols());
  output.topRows(A.rows()) = x_hat;
  output.bottomRows(S.rows()) = y_hat;
  return output;
}

template <typename Solver>
inline bool BlockSymmetric<Solver>::
operator==(const BlockSymmetric &rhs) const {
  return (A == rhs.A && Li_B == rhs.Li_B && S == rhs.S);
}

template <typename Solver>
template <typename Archive>
inline void BlockSymmetric<Solver>::serialize(Archive &archive,
                                              const std::uint32_t) {
  archive(cereal::make_nvp("A", A), cereal::make_nvp("Li_B", Li_B),
          cereal::make_nvp("S", S));
}

template <typename Solver>
inline Eigen::Index BlockSymmetric<Solver>::rows() const {
  return A.rows() + S.rows();
}

template <typename Solver>
inline Eigen::Index BlockSymmetric<Solver>::cols() const {
  return A.cols() + S.cols();
}

template <typename Solver>
BlockSymmetric<Solver> build_block_symmetric(const Solver &A,
                                             const Eigen::MatrixXd &B,
                                             const Eigen::SerializableLDLT &S) {
  return BlockSymmetric<Solver>(A, B, S);
}

template <typename Solver>
BlockSymmetric<Solver> build_block_symmetric(const Solver &A,
                                             const Eigen::MatrixXd &B,
                                             const Eigen::MatrixXd &C) {
  return BlockSymmetric<Solver>(A, B, C);
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_UTILS_BLOCK_UTILS_H_ */
