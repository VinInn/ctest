#include <Eigen/Dense>

    constexpr int SampleVectorSize = 10;
    constexpr int FullSampleVectorSize = 19;
    constexpr int PulseVectorSize = 12;
    constexpr int NGains = 3;

    using data_type = double;

    typedef Eigen::Matrix<data_type, SampleVectorSize, 1> SampleVector;
    typedef Eigen::Matrix<data_type, SampleVectorSize, SampleVectorSize> SampleMatrix;
    typedef Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic, 0, PulseVectorSize, PulseVectorSize> PulseMatrix;

    typedef Eigen::LLT<SampleMatrix> SampleDecompLLT;
    typedef Eigen::LLT<PulseMatrix> PulseDecompLLT;
    typedef Eigen::LDLT<PulseMatrix> PulseDecompLDLT;



    void eigen_solve_submatrix(SampleMatrix& mat, SampleVector& invec, SampleVector& outvec, unsigned NP) {
      using namespace Eigen;
      switch (NP) {  // pulse matrix is always square.
        case 10: {
          Matrix<SampleMatrix::Scalar, 10, 10> temp = mat.topLeftCorner<10, 10>();
          outvec.head<10>() = temp.ldlt().solve(invec.head<10>());
          break;
        }
        case 9: {
          Matrix<SampleMatrix::Scalar, 9, 9> temp = mat.topLeftCorner<9, 9>();
          outvec.head<9>() = temp.ldlt().solve(invec.head<9>());
          break;
        }
        case 8: {
          Matrix<SampleMatrix::Scalar, 8, 8> temp = mat.topLeftCorner<8, 8>();
          outvec.head<8>() = temp.ldlt().solve(invec.head<8>());
          break;
        }
        case 7: {
          Matrix<SampleMatrix::Scalar, 7, 7> temp = mat.topLeftCorner<7, 7>();
          outvec.head<7>() = temp.ldlt().solve(invec.head<7>());
          break;
        }
        case 6: {
          Matrix<SampleMatrix::Scalar, 6, 6> temp = mat.topLeftCorner<6, 6>();
          outvec.head<6>() = temp.ldlt().solve(invec.head<6>());
          break;
        }
        case 5: {
          Matrix<SampleMatrix::Scalar, 5, 5> temp = mat.topLeftCorner<5, 5>();
          outvec.head<5>() = temp.ldlt().solve(invec.head<5>());
          break;
        }
        case 4: {
          Matrix<SampleMatrix::Scalar, 4, 4> temp = mat.topLeftCorner<4, 4>();
          outvec.head<4>() = temp.ldlt().solve(invec.head<4>());
          break;
        }
        default:
           return;
    }
}

__device__ bool inplace_fnnls(SampleMatrix const& A,
                              SampleVector const& b,
                              PulseMatrix & pulse_matrix);


    __device__ __forceinline__ SampleVector::Scalar compute_chi2(SampleDecompLLT& covariance_decomposition,
                                                                 PulseMatrix const& pulse_matrix,
                                                                 SampleVector const& amplitudes,
                                                                 SampleVector const& samples) {
      return covariance_decomposition.matrixL().solve(pulse_matrix * amplitudes - samples).squaredNorm();
    }   

__global__
void go(SampleMatrix * inverse_cov, PulseMatrix * pulse_matrix, SampleVector * samples,  SampleVector * amplitudes, int max_iterations)  {

    int iter=0;
    while (true) {
          if (iter >= max_iterations)
            break;

    SampleDecompLLT covariance_decomposition;
    covariance_decomposition.compute(inverse_cov[threadIdx.x]);

    SampleMatrix A = covariance_decomposition.matrixL().solve(pulse_matrix[threadIdx.x]);
    SampleVector b = covariance_decomposition.matrixL().solve(samples[threadIdx.x]);

     // samples[threadIdx.x] = A*b;
     auto t = inplace_fnnls(A,b,pulse_matrix[threadIdx.x]);

     if (!t) break;

     auto chi2 = compute_chi2(covariance_decomposition,pulse_matrix[threadIdx.x],amplitudes[threadIdx.x],samples[threadIdx.x]);
     
     if (chi2<1.e-3) break;
     ++iter;
   } 

}
