#include "inplace_fnnls.h"

    __device__ __forceinline__ SampleVector::Scalar compute_chi2(SampleDecompLLT& covariance_decomposition,
                                                                 PulseMatrix const& pulse_matrix,
                                                                 SampleVector const& amplitudes,
                                                                 SampleVector const& samples) {
      return covariance_decomposition.matrixL().solve(pulse_matrix * amplitudes - samples).squaredNorm();
    }


void goCC(SampleMatrix * inverse_cov, PulseMatrix * pulse_matrix, SampleVector * samples,  SampleVector * amplitudes, BXVector * bxs, int max_iterations)  {

    int iter=0;
    while (true) {
          if (iter >= max_iterations)
            break;

    SampleDecompLLT covariance_decomposition;
    covariance_decomposition.compute(inverse_cov[threadIdx.x]);

    SampleMatrix A = covariance_decomposition.matrixL().solve(pulse_matrix[threadIdx.x]);
    SampleVector b = covariance_decomposition.matrixL().solve(samples[threadIdx.x]);

     // samples[threadIdx.x] = A*b;
     int npassive=0;
     auto t = inplace_fnnls(A,b, amplitudes[threadIdx.x], npassive,bxs[threadIdx.x],pulse_matrix[threadIdx.x]);

     if (!t) break;

     auto chi2 = compute_chi2(covariance_decomposition,pulse_matrix[threadIdx.x],amplitudes[threadIdx.x],samples[threadIdx.x]);
     
     if (chi2<1.e-3) break;
     ++iter;
   } 

}
