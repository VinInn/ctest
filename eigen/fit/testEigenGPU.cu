#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "RiemannFit.h"

#include "test_common.h"
#include "../../cuda/cudaCheck.h"

using namespace Eigen;

namespace Rfit {
  constexpr uint32_t maxNumberOfTracks() { return 10000; }
  constexpr uint32_t stride() { return maxNumberOfTracks();}
  using Matrix3x4d = Eigen::Matrix<double,3,4>;
  using Map3x4d = Eigen::Map<Matrix3x4d,0,Eigen::Stride<3*stride(),stride()> >;
  using Map4d = Eigen::Map<Vector4d,0,Eigen::InnerStride<stride()> >;
}

__global__
void kernelFastFit(double * __restrict__ phits, double * __restrict__ presults) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  Rfit::Map3x4d hits(phits+i,3,4);
  Rfit::Map4d result(presults+i,4);
  Rfit::Fast_fit(hits,  result);
}

__global__
void kernelCircleFit(double * __restrict__ phits,
    Rfit::Matrix3Nd * hits_cov, 
    double * __restrict__ pfast_fit_input, 
    double B,
    Rfit::circle_fit * circle_fit_resultsGPU) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  Rfit::Map3x4d hits(phits+i,3,4);
  Rfit::Map4d   fast_fit_input(pfast_fit_input+i,4);

  u_int n = hits.cols();
  Rfit::VectorNd rad = (hits.block(0, 0, 2, n).colwise().norm());

#if TEST_DEBUG
if (0==i) {
  printf("hits %f, %f\n", hits.block(0,0,2,n)(0,0), hits.block(0,0,2,n)(0,1));
  printf("hits %f, %f\n", hits.block(0,0,2,n)(1,0), hits.block(0,0,2,n)(1,1));
  printf("fast_fit_input(0): %f\n", fast_fit_input(0));
  printf("fast_fit_input(1): %f\n", fast_fit_input(1));
  printf("fast_fit_input(2): %f\n", fast_fit_input(2));
  printf("fast_fit_input(3): %f\n", fast_fit_input(3));
  printf("rad(0,0): %f\n", rad(0,0));
  printf("rad(1,1): %f\n", rad(1,1));
  printf("rad(2,2): %f\n", rad(2,2));
  printf("hits_cov(0,0): %f\n", (*hits_cov)(0,0));
  printf("hits_cov(1,1): %f\n", (*hits_cov)(1,1));
  printf("hits_cov(2,2): %f\n", (*hits_cov)(2,2));
  printf("hits_cov(11,11): %f\n", (*hits_cov)(11,11));
  printf("B: %f\n", B);
}
#endif
  circle_fit_resultsGPU[i] =
    Rfit::Circle_fit(hits.block(0,0,2,n), hits_cov[i].block(0, 0, 2 * n, 2 * n),
      fast_fit_input, rad, B, true);
#if TEST_DEBUG
if (0==i) {
  printf("Circle param %f,%f,%f\n",circle_fit_resultsGPU[i].par(0),circle_fit_resultsGPU[i].par(1),circle_fit_resultsGPU[i].par(2));
}
#endif
}

__global__
void kernelLineFit(double * __restrict__ phits,
                   Rfit::Matrix3Nd * hits_cov,
                   Rfit::circle_fit * circle_fit,
                   double * __restrict__ pfast_fit,
                   Rfit::line_fit * line_fit)
{
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  Rfit::Map3x4d hits(phits+i,3,4);
  Rfit::Map4d   fast_fit(pfast_fit+i,4);

  line_fit[i] = Rfit::Line_fit(hits, hits_cov[i], circle_fit[i], fast_fit, true);
}

template<typename M3x4, typename M3N>
__device__ __host__
void fillHitsAndHitsCov(M3x4 & hits, M3N & hits_cov) {
  hits << 1.98645, 4.72598, 7.65632, 11.3151,
          2.18002, 4.88864, 7.75845, 11.3134,
          2.46338, 6.99838,  11.808,  17.793;
  hits_cov(0,0) = 7.14652e-06;
  hits_cov(1,1) = 2.15789e-06;
  hits_cov(2,2) = 1.63328e-06;
  hits_cov(3,3) = 6.27919e-06;
  hits_cov(4,4) = 6.10348e-06;
  hits_cov(5,5) = 2.08211e-06;
  hits_cov(6,6) = 1.61672e-06;
  hits_cov(7,7) = 6.28081e-06;
  hits_cov(8,8) = 5.184e-05;
  hits_cov(9,9) = 1.444e-05;
  hits_cov(10,10) = 6.25e-06;
  hits_cov(11,11) = 3.136e-05;
  hits_cov(0,4) = hits_cov(4,0) = -5.60077e-06;
  hits_cov(1,5) = hits_cov(5,1) = -1.11936e-06;
  hits_cov(2,6) = hits_cov(6,2) = -6.24945e-07;
  hits_cov(3,7) = hits_cov(7,3) = -5.28e-06;
}

__global__
void kernelFillHitsAndHitsCov(double * __restrict__ phits,
  Rfit::Matrix3Nd * hits_cov) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  Rfit::Map3x4d hits(phits+i,3,4);
  hits_cov[i] = MatrixXd::Zero(12,12);
  fillHitsAndHitsCov(hits,hits_cov[i]);
}

void testFit() {
  constexpr double B = 0.0113921;
  Rfit::Matrix3xNd hits(3,4);
  Rfit::Matrix3Nd hits_cov = MatrixXd::Zero(12,12);
  double * hitsGPU = nullptr;;
  double * hits_covGPU = nullptr;
  double * fast_fit_resultsGPU = nullptr;
  double * fast_fit_resultsGPUret = new double[Rfit::maxNumberOfTracks()*sizeof(Vector4d)];
  Rfit::circle_fit * circle_fit_resultsGPU = nullptr;
  Rfit::circle_fit * circle_fit_resultsGPUret = new Rfit::circle_fit();
  Rfit::line_fit * line_fit_resultsGPU = nullptr;

  fillHitsAndHitsCov(hits, hits_cov);
  
  std::cout << "Generated hits:\n" << hits << std::endl;
  std::cout << "Generated cov:\n" << hits_cov << std::endl;

  // FAST_FIT_CPU
  Vector4d fast_fit_results; Rfit::Fast_fit(hits, fast_fit_results);
  std::cout << "Fitted values (FastFit, [X0, Y0, R, tan(theta)]):\n" << fast_fit_results << std::endl;

  // for timing    purposes we fit    4096 tracks
  constexpr uint32_t Ntracks = 4096;
  cudaCheck(cudaMalloc((void **)&hitsGPU, Rfit::maxNumberOfTracks()*sizeof(Rfit::Matrix3xNd(3,4))));
  cudaCheck(cudaMalloc((void **)&hits_covGPU, Rfit::maxNumberOfTracks()*sizeof(Rfit::Matrix3Nd(12,12))));
  cudaMalloc((void**)&fast_fit_resultsGPU, Rfit::maxNumberOfTracks()*sizeof(Vector4d));
  cudaCheck(cudaMalloc((void **)&line_fit_resultsGPU, Rfit::maxNumberOfTracks()*sizeof(Rfit::line_fit)));
  cudaCheck(cudaMalloc((void **)&circle_fit_resultsGPU, Rfit::maxNumberOfTracks()*sizeof(Rfit::circle_fit)));


  kernelFillHitsAndHitsCov<<<Ntracks/64, 64>>>(hitsGPU,(Rfit::Matrix3Nd *)hits_covGPU);

  // FAST_FIT GPU
  kernelFastFit<<<Ntracks/64, 64>>>(hitsGPU, fast_fit_resultsGPU);
  cudaDeviceSynchronize();
  
  cudaMemcpy(fast_fit_resultsGPUret, fast_fit_resultsGPU, Rfit::maxNumberOfTracks()*sizeof(Vector4d), cudaMemcpyDeviceToHost);
  Rfit::Map4d fast_fit(fast_fit_resultsGPUret+10,4);
  std::cout << "Fitted values (FastFit, [X0, Y0, R, tan(theta)]): GPU\n" << fast_fit << std::endl;
  assert(isEqualFuzzy(fast_fit_results, fast_fit));

  // CIRCLE_FIT CPU
  u_int n = hits.cols();
  Rfit::VectorNd rad = (hits.block(0, 0, 2, n).colwise().norm());

  Rfit::circle_fit circle_fit_results = Rfit::Circle_fit(hits.block(0, 0, 2, n),
      hits_cov.block(0, 0, 2 * n, 2 * n),
      fast_fit_results, rad, B, true);
  std::cout << "Fitted values (CircleFit):\n" << circle_fit_results.par << std::endl;

  // CIRCLE_FIT GPU

  kernelCircleFit<<<Ntracks/64, 64>>>(hitsGPU, (Rfit::Matrix3Nd *)hits_covGPU,
      fast_fit_resultsGPU, B, circle_fit_resultsGPU);
  cudaDeviceSynchronize();

  cudaMemcpy(circle_fit_resultsGPUret, circle_fit_resultsGPU,
      sizeof(Rfit::circle_fit), cudaMemcpyDeviceToHost);
  std::cout << "Fitted values (CircleFit) GPU:\n" << circle_fit_resultsGPUret->par << std::endl;
  assert(isEqualFuzzy(circle_fit_results.par, circle_fit_resultsGPUret->par));

  // LINE_FIT CPU
  Rfit::line_fit line_fit_results = Rfit::Line_fit(hits, hits_cov, circle_fit_results, fast_fit_results, true);
  std::cout << "Fitted values (LineFit):\n" << line_fit_results.par << std::endl;

  // LINE_FIT GPU
  Rfit::line_fit * line_fit_resultsGPUret = new Rfit::line_fit();

  kernelLineFit<<<Ntracks/64, 64>>>(hitsGPU, (Rfit::Matrix3Nd *)hits_covGPU, circle_fit_resultsGPU, fast_fit_resultsGPU, line_fit_resultsGPU);
  cudaDeviceSynchronize();

  cudaMemcpy(line_fit_resultsGPUret, line_fit_resultsGPU, sizeof(Rfit::line_fit), cudaMemcpyDeviceToHost);
  std::cout << "Fitted values (LineFit) GPU:\n" << line_fit_resultsGPUret->par << std::endl;
  assert(isEqualFuzzy(line_fit_results.par, line_fit_resultsGPUret->par));

  std::cout << "Fitted cov (CircleFit) CPU:\n" << circle_fit_results.cov << std::endl;
  std::cout << "Fitted cov (LineFit): CPU\n" << line_fit_results.cov << std::endl;
  std::cout << "Fitted cov (CircleFit) GPU:\n" << circle_fit_resultsGPUret->cov << std::endl;
  std::cout << "Fitted cov (LineFit): GPU\n" << line_fit_resultsGPUret->cov << std::endl;

}

int main (int argc, char * argv[]) {
  testFit();
  std::cout << "TEST FIT, NO ERRORS" << std::endl;

  return 0;
}

