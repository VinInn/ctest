// some  not needed as done by cuda runtime...
#ifndef __CUDA_RUNTIME_H__
#define __host__
#define __device__
#define __global__
#define __shared__
#define __forceinline__
#endif

#ifndef __CUDA_RUNTIME_H__
  struct dim3 {
    int x, y, z;
  };
  const dim3 threadIdx = {0, 0, 0};
#endif

// make sure function are inlined to avoid multiple definition
#ifndef __CUDA_ARCH__
#undef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif

#define EIGEN_NO_DEBUG
#include <Eigen/Dense>

    constexpr int SampleVectorSize = 10;
    constexpr int FullSampleVectorSize = 19;
    constexpr int PulseVectorSize = 12;
    constexpr int NGains = 3;

    using data_type = double;

    typedef Eigen::Matrix<data_type, SampleVectorSize, 1> SampleVector;
    typedef Eigen::Matrix<data_type, SampleVectorSize, SampleVectorSize> SampleMatrix;
    typedef Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic, 0, PulseVectorSize, PulseVectorSize> PulseMatrix;
    typedef Eigen::Matrix<char, SampleVectorSize, 1> BXVector;

    typedef Eigen::LLT<SampleMatrix> SampleDecompLLT;
    typedef Eigen::LLT<PulseMatrix> PulseDecompLLT;
    typedef Eigen::LDLT<PulseMatrix> PulseDecompLDLT;

    using matrix_t = SampleMatrix;
    using vector_t = SampleVector;

    __device__ __forceinline__ bool inplace_fnnls(matrix_t const& A,
                                  vector_t const& b,
                                  vector_t& x,
                                  int& npassive,
                                  BXVector& activeBXs,
                                  PulseMatrix& pulse_matrix,
                                  const double eps = 1e-11,
                                  const unsigned int max_iterations = 500) {
      matrix_t AtA = A.transpose() * A;
      vector_t Atb = A.transpose() * b;
      vector_t s;
      vector_t w;

      // main loop
      Eigen::Index w_max_idx_prev = 0;
      matrix_t::Scalar w_max_prev = 0;
      double eps_to_use = eps;

      int iter = 0;
      while (true) {
        if (iter > 0 || npassive == 0) {
          const auto nActive = vector_t::RowsAtCompileTime - npassive;
          if (!nActive)
            break;

          w.tail(nActive) = Atb.tail(nActive) - (AtA * x).tail(nActive);

          // get the index of w that gives the maximum gain
          Eigen::Index w_max_idx;
          const auto max_w = w.tail(nActive).maxCoeff(&w_max_idx);

          // check for convergence
          if (max_w < eps_to_use || (w_max_idx == w_max_idx_prev && max_w == w_max_prev))
            break;

          // worst case
          if (iter >= 500)
            break;

          w_max_prev = max_w;
          w_max_idx_prev = w_max_idx;

          // need to translate the index into the right part of the vector
          w_max_idx += npassive;

          // swap AtA to avoid copy
          AtA.col(npassive).swap(AtA.col(w_max_idx));
          AtA.row(npassive).swap(AtA.row(w_max_idx));
          // swap Atb to match with AtA
          Eigen::numext::swap(Atb.coeffRef(npassive), Atb.coeffRef(w_max_idx));
          Eigen::numext::swap(x.coeffRef(npassive), x.coeffRef(w_max_idx));
          Eigen::numext::swap(activeBXs.coeffRef(npassive), activeBXs.coeffRef(w_max_idx));
          pulse_matrix.col(npassive).swap(pulse_matrix.col(w_max_idx));

          ++npassive;
        }

        // inner loop
        while (true) {
          if (npassive == 0)
            break;

          s.head(npassive) = AtA.topLeftCorner(npassive, npassive).llt().solve(Atb.head(npassive));

          // if all coefficients are positive, done for this iteration
          if (s.head(npassive).minCoeff() > 0.) {
            x.head(npassive) = s.head(npassive);
            break;
          }

          auto alpha = std::numeric_limits<double>::max();
          Eigen::Index alpha_idx = 0;

#pragma unroll
          for (auto i = 0; i < npassive; ++i) {
            if (s[i] <= 0.) {
              auto const ratio = x[i] / (x[i] - s[i]);
              if (ratio < alpha) {
                alpha = ratio;
                alpha_idx = i;
              }
            }
          }

          /*
      if (std::numeric_limits<double>::max() == alpha) {
        x.head(npassive) = s.head(npassive);
        break;
      }*/

          x.head(npassive) += alpha * (s.head(npassive) - x.head(npassive));
          x[alpha_idx] = 0;
          --npassive;

          AtA.col(npassive).swap(AtA.col(alpha_idx));
          AtA.row(npassive).swap(AtA.row(alpha_idx));
          // swap Atb to match with AtA
          Eigen::numext::swap(Atb.coeffRef(npassive), Atb.coeffRef(alpha_idx));
          Eigen::numext::swap(x.coeffRef(npassive), x.coeffRef(alpha_idx));
          Eigen::numext::swap(activeBXs.coeffRef(npassive), activeBXs.coeffRef(alpha_idx));
          pulse_matrix.col(npassive).swap(pulse_matrix.col(alpha_idx));
        }

        // TODO as in cpu NNLS version
        iter++;
        if (iter % 16 == 0)
          eps_to_use *= 2;
      }

      return true;
    }

