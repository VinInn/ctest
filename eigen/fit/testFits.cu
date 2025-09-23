#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#ifdef USE_BL
#include "BrokenLine.h"
#else
#include "Rfit.h"
#endif


#include "test_common.h"
#include "../../cuda/cudaCheck.h"
#include "../../cuda/exitSansCUDADevices.h"

//#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
//#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"
//#include "RecoPixelVertexing/PixelTrackFitting/interface/Rfit.h"

namespace Rfit {
  constexpr uint32_t maxNumberOfTracks() { return 17*1024; }
  constexpr uint32_t stride() { return maxNumberOfTracks();}

  // for computed errors
  using Vector5dd = Eigen::Vector<double, 5>;
  // for true trajectory
  using Vector6dd = Eigen::Vector<double, 6>;

  // hits
//  template<int N>
//  using Matrix3xNd = Eigen::Matrix<FF,3,N>;
  template<int N>
  using Map3xNd = Eigen::Map<Matrix3xNd<N>,0,Eigen::Stride<3*stride(),stride()> >;
  // errors
  template<int N>
  using Matrix6xNf = Eigen::Matrix<float,6,N>;
  template<int N>
  using Map6xNf = Eigen::Map<Matrix6xNf<N>,0,Eigen::Stride<6*stride(),stride()> >;
  // fast fit
  using Map4d = Eigen::Map<Rfit::Vector4d,0,Eigen::InnerStride<stride()> >;
}

// using namespace Eigen;


template<int N>
__global__
void kernelPrintSizes(Rfit::FF * __restrict__ phits,
                      float * __restrict__ phits_ge
		      ) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  Rfit::Map3xNd<N> hits(phits+i,3,4);
  Rfit::Map6xNf<N> hits_ge(phits_ge+i,6,4);
  if (i!=0) return;
  printf("GPU sizes %lu %lu %lu %lu %lu\n",sizeof(hits[i]),sizeof(hits_ge[i]),
	 sizeof(Rfit::Vector4d),sizeof(Rfit::line_fit),sizeof(Rfit::circle_fit));
}


template<int N>
__global__
void kernelFastFit(Rfit::FF * __restrict__ phits, Rfit::FF * __restrict__ presults, int nt, int64_t * tg) {

  __shared__ uint64_t gstart;
  if (threadIdx.x==0) gstart = clock64();
  __syncthreads();

  int first = blockIdx.x*blockDim.x + threadIdx.x;

  for (int i=first; i<nt; i+=gridDim.x * blockDim.x) {

  Rfit::Map3xNd<N> hits(phits+i,3,N);
  Rfit::Map4d result(presults+i,4);

#ifdef USE_BL
  BrokenLine::BL_Fast_fit(hits, result);
#else
  Rfit::Fast_fit(hits,  result);
#endif

  }
  if (threadIdx.x==0) tg[blockIdx.x] = clock64() -gstart;
}

#ifdef USE_BL

template<int N>
__global__
void kernelBrokenLineFit(Rfit::FF * __restrict__ phits,
			 float * __restrict__ phits_ge, 
			 Rfit::FF * __restrict__ pfast_fit_input, 
			 Rfit::Float B,
			 Rfit::circle_fit * circle_fit,
			 Rfit::line_fit * line_fit,
                          int nt,
                         int64_t * tg
			 ) {

  __shared__ uint64_t gstart;
  if (threadIdx.x==0) gstart = clock64();
  __syncthreads();

  int first = blockIdx.x*blockDim.x + threadIdx.x;

  for (int i=first; i<nt; i+=gridDim.x * blockDim.x) {


  Rfit::Map3xNd<N> hits(phits+i,3,N);
  Rfit::Map4d   fast_fit_input(pfast_fit_input+i,4);
  Rfit::Map6xNf<N> hits_ge(phits_ge+i,6,N);
  
  BrokenLine::PreparedBrokenLineData<N> data;
  Rfit::Matrix3d Jacob;
  
  auto & line_fit_results = line_fit[i];
  auto & circle_fit_results = circle_fit[i];
  
  BrokenLine::prepareBrokenLineData(hits,fast_fit_input,B,data);
  BrokenLine::BL_Line_fit(hits_ge,fast_fit_input,B,data,line_fit_results);
  BrokenLine::BL_Circle_fit(hits,hits_ge,fast_fit_input,B,data,circle_fit_results);
  Jacob << 1.,0,0,
    0,1.,0,
    0,0,-B/std::copysign(Rfit::sqr(circle_fit_results.par(2)),circle_fit_results.par(2));
  circle_fit_results.par(2)=B/fabs(circle_fit_results.par(2));
  circle_fit_results.cov=Jacob*circle_fit_results.cov*Jacob.transpose();
  }
  
__syncthreads();
  if (threadIdx.x==0) tg[blockIdx.x] = clock64() -gstart;


#ifdef TEST_DEBUG
if (0==i) {
  printf("Circle param %f,%f,%f\n",circle_fit[i].par(0),circle_fit[i].par(1),circle_fit[i].par(2));
 }
#endif
}

#else

template<int N>
__global__
void kernelCircleFit(Rfit::FF * __restrict__ phits,
    float * __restrict__ phits_ge, 
    Rfit::FF * __restrict__ pfast_fit_input, 
    Rfit::Float B,
    Rfit::circle_fit * circle_fit_resultsGPU) {

  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  Rfit::Map3xNd<N> hits(phits+i,3,N);
  Rfit::Map4d   fast_fit_input(pfast_fit_input+i,4);
  Rfit::Map6xNf<N> hits_ge(phits_ge+i,6,N);

  constexpr auto n = N;

  Rfit::VectorNd<N> rad = (hits.block(0, 0, 2, n).colwise().norm());
  Rfit::Matrix2Nd<N> hits_cov =  Rfit::MatrixXd::Zero(2 * n, 2 * n);
  Rfit::loadCovariance2D(hits_ge,hits_cov);
  
#ifdef TEST_DEBUG
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
    Rfit::Circle_fit(hits.block(0,0,2,n), hits_cov,
      fast_fit_input, rad, B, true);
#ifdef TEST_DEBUG
if (0==i) {
  printf("Circle param %f,%f,%f\n",circle_fit_resultsGPU[i].par(0),circle_fit_resultsGPU[i].par(1),circle_fit_resultsGPU[i].par(2));
}
#endif
}

template<int N>
__global__
void kernelLineFit(Rfit::FF * __restrict__ phits,
		   float * __restrict__ phits_ge,
                   Rfit::Float B,
                   Rfit::circle_fit * circle_fit,
                   Rfit::FF * __restrict__ pfast_fit_input,
                   Rfit::line_fit * line_fit)
{
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  Rfit::Map3xNd<N> hits(phits+i,3,N);
  Rfit::Map4d   fast_fit_input(pfast_fit_input+i,4);
  Rfit::Map6xNf<N> hits_ge(phits_ge+i,6,N);
  line_fit[i] = Rfit::Line_fit(hits, hits_ge, circle_fit[i], fast_fit_input, B, true);
}
#endif


////////////////

Rfit::Vector5dd True_par(const Rfit::Vector6dd & gen_par, int charge, double B_field) {
  Rfit::Vector5dd true_par;
  constexpr double pi = M_PI;
  const double x0 = gen_par(0) + gen_par(4) * cos(gen_par(3) * pi / 180);
  const double y0 = gen_par(1) + gen_par(4) * sin(gen_par(3) * pi / 180);
  Rfit::circle_fit circle;
  circle.par << x0, y0, gen_par(4);
  circle.q = charge;
  Rfit::par_uvrtopak(circle, B_field, false);
  for (int i = 0; i<3; ++i) true_par[i] = toDouble(circle.par[i]);
  true_par(3) = 1 / tan(gen_par(5) * pi / 180);
  const int dir = ((gen_par(0) - cos(true_par(0) - pi / 2) * true_par(1)) * (gen_par(1) - y0) -
                       (gen_par(1) - sin(true_par(0) - pi / 2) * true_par(1)) * (gen_par(0) - x0) >
                   0)
                      ? -1
                      : 1;
  true_par(4) = gen_par(2) + 1 / tan(gen_par(5) * pi / 180) * dir * 2.f *
                                 asin(sqrt(Rfit::sqr((gen_par(0) - cos(true_par(0) - pi / 2) * true_par(1))) +
                                           Rfit::sqr((gen_par(1) - sin(true_par(0) - pi / 2) * true_par(1)))) /
                                      (2.f * gen_par(4))) *
                                 gen_par(4);
  return true_par;
}

Rfit::Vector6dd New_par(const Rfit::Vector6dd& gen_par, const int& charge, const double& B_field) {
  constexpr double pi = M_PI;
  Rfit::Vector6dd new_par;
  new_par.block(0, 0, 3, 1) = gen_par.block(0, 0, 3, 1);
  new_par(3) = gen_par(3) - charge * 90;
  new_par(4) = gen_par(4) / B_field;
  //  new_par(5) = atan(sinh(gen_par(5))) * 180 / pi;
  new_par(5) = 2. * atan(exp(-gen_par(5))) * 180 / pi;
  return new_par;
}



template<typename M3xN, typename M6xN>
__device__ __host__
void Hits_cov(M6xN& V,
              const unsigned int& i,
              M3xN & hits,
              const Rfit::Vector5dd& err,
              bool isbarrel) {
  if (isbarrel) {
    double R2 = Rfit::sqr(hits(0, i)) + Rfit::sqr(hits(1, i));
    V.col(i)[0] = (Rfit::sqr(err[1]) * Rfit::sqr(hits(1, i)) +
                   Rfit::sqr(err[0]) * Rfit::sqr(hits(0, i))) /
                  R2;
    V.col(i)[2] = (Rfit::sqr(err[1]) * Rfit::sqr(hits(0, i)) +
                   Rfit::sqr(err[0]) * Rfit::sqr(hits(1, i))) /
                  R2;
    V.col(i)[1] = (Rfit::sqr(err[0]) - Rfit::sqr(err[1])) * hits(1, i) * hits(0, i) / R2;
    V.col(i)[5] = Rfit::sqr(err[2]);
  } else {
    V.col(i)[0] = Rfit::sqr(err[3]);
    V.col(i)[2] = Rfit::sqr(err[3]);
    V.col(i)[5] = Rfit::sqr(err[4]);
  }
}




template<typename M3xN, typename M6xN>
__device__ __host__
void fillHitsAndHitsCov(Rfit::Vector6dd const & gen_par,  M3xN & hits, M6xN & hits_ge) {

  constexpr double pi = M_PI;

  constexpr uint32_t N = M3xN::ColsAtCompileTime;

  constexpr double rad[8] = {2.95, 6.8, 10.9, 16., 3.1, 7., 11., 16.2};
  // constexpr double R_err[8] = {5./10000, 5./10000, 5./10000, 5./10000, 5./10000,
  // 5./10000, 5./10000, 5./10000};  constexpr double Rp_err[8] = {35./10000, 18./10000,
  // 15./10000, 34./10000, 35./10000, 18./10000, 15./10000, 34./10000};  constexpr double z_err[8] =
  // {72./10000, 38./10000, 25./10000, 56./10000, 72./10000, 38./10000, 25./10000, 56./10000};
  constexpr double R_err[8] = {
      10. / 10000, 10. / 10000, 10. / 10000, 10. / 10000, 10. / 10000, 10. / 10000, 10. / 10000, 10. / 10000};
  constexpr double Rp_err[8] = {
      35. / 10000, 18. / 10000, 15. / 10000, 34. / 10000, 35. / 10000, 18. / 10000, 15. / 10000, 34. / 10000};
  constexpr double z_err[8] = {
      72. / 10000, 38. / 10000, 25. / 10000, 56. / 10000, 72. / 10000, 38. / 10000, 25. / 10000, 56. / 10000};
  const double x2 = gen_par(0) + gen_par(4) * std::cos(gen_par(3) * (pi / 180.));
  const double y2 = gen_par(1) + gen_par(4) * std::sin(gen_par(3) * (pi / 180.) );
  const double alpha = std::atan2(y2, x2);


 for (unsigned int i = 0; i < N; ++i) {
    const double a = gen_par(4);
    const double b = rad[i];
    const double c = std::sqrt(Rfit::sqr(x2) + Rfit::sqr(y2));
    const double beta = std::acos((Rfit::sqr(a) - Rfit::sqr(b) - Rfit::sqr(c)) / (-2. * b * c));
    const double gamma = alpha + beta;
    hits(0, i) = rad[i] * std::cos(gamma);
    hits(1, i) = rad[i] * std::sin(gamma);
    hits(2, i) =
        gen_par(2) +
        1. / std::tan(gen_par(5) * (pi / 180)) * 2. *
            std::asin(sqrt(Rfit::sqr((gen_par(0) - hits(0, i))) + Rfit::sqr((gen_par(1) - hits(1, i)))) /
                 (2. * gen_par(4))) *
            gen_par(4);
    // isbarrel(i) = ??
    Rfit::Vector5dd err;
    err << R_err[i], Rp_err[i], z_err[i], 0, 0;
    // smearing(err, true, gen.hits(0, i), gen.hits(1, i), gen.hits(2, i));
    Hits_cov(hits_ge, i, hits, err, true);
  }


}


template<int N>
__global__
void kernelFillHitsAndHitsCov(Rfit::Vector6dd * gen_par,  Rfit::FF * __restrict__ phits,
  float * phits_ge) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  Rfit::Map3xNd<N> hits(phits+i,3,N);
  Rfit::Map6xNf<N> hits_ge(phits_ge+i,6,N);
  hits_ge = Eigen::MatrixXf::Zero(6,N);
  fillHitsAndHitsCov(gen_par[0], hits,hits_ge);
}

template<int N>
void testFit() {

  // for timing    purposes we fit   16K tracks
  constexpr uint32_t Ntracks = 16*1024;

  constexpr Rfit::Float B = 0.0113921;
  Rfit::Matrix3xNd<N> hits;
  Rfit::Matrix6xNf<N> hits_ge = Eigen::MatrixXf::Zero(6,N);

  Rfit::Vector6dd gen_par;
//
  std::cout << "_________________________________________________________________________\n";
  std::cout << "n x(cm) y(cm) z(cm) phi(grad) R(Gev/c) eta" << std::endl;
  gen_par(0) = -0.1;  // x
  gen_par(1) = 0.1;   // y
  gen_par(2) = -1.;   // z
  gen_par(3) = 45.;   // phi
  gen_par(4) = 500000.;   // R (p_t)
  gen_par(5) = 0.5;    // eta

  std::cout << gen_par << std::endl;

  gen_par = New_par(gen_par, 1, B);

  std::cout << gen_par << std::endl;
  auto true_par = True_par(gen_par, 1, B);

  std::cout << "\nTrue parameters: "
            << "phi: " << true_par(0) << " "
            << "dxy: " << true_par(1) << " "
            << "pt: " << true_par(2) << " "
            << "CotT: " << true_par(3) << " "
            << "Zip: " << true_par(4) << " " << std::endl;

  std::cout << std::endl;

//
#ifndef  NOGPU
  Rfit::FF * hitsGPU = nullptr;
  float * hits_geGPU = nullptr;
  Rfit::FF * fast_fit_resultsGPU = nullptr;
  Rfit::FF * fast_fit_resultsGPUret = new Rfit::FF[Rfit::maxNumberOfTracks()*sizeof(Rfit::Vector4d)];
  Rfit::circle_fit * circle_fit_resultsGPU = nullptr;
  Rfit::circle_fit * circle_fit_resultsGPUret = new Rfit::circle_fit();
  Rfit::line_fit * line_fit_resultsGPU = nullptr;
  Rfit::line_fit * line_fit_resultsGPUret = new Rfit::line_fit();
  Rfit::Vector6dd * gen_parGPU = nullptr;
#endif


  fillHitsAndHitsCov(gen_par, hits, hits_ge);

  std::cout << "sizes " << N << ' '
	    << sizeof(hits) << ' ' << sizeof(hits_ge)
	    << ' ' << sizeof(Rfit::Vector4d) 
	    << ' ' << sizeof(Rfit::line_fit) 
            << ' ' << sizeof(Rfit::circle_fit)
            << std::endl;
  
  std::cout << "Generated hits:\n" << hits << std::endl;
  std::cout << "Generated cov:\n" << hits_ge << std::endl;

 Rfit::Vector4d fast_fit_results[Ntracks];
 uint32_t kk=0;
#ifdef  NOGPU
for (uint32_t k=0;k<32*Ntracks; ++k, kk=k/32) 
#endif
{
assert(kk<Ntracks);
  // FAST_FIT_CPU
#ifdef USE_BL
  BrokenLine::BL_Fast_fit(hits, fast_fit_results[kk]);
#else
  Rfit::Fast_fit(hits, fast_fit_results[kk]);
#endif
}

  std::cout << "Fitted values (FastFit, [X0, Y0, R, tan(theta)]):\n" << fast_fit_results[0] << std::endl;


#ifndef  NOGPU
  cudaCheck(cudaMalloc(&hitsGPU, Rfit::maxNumberOfTracks()*sizeof(Rfit::Matrix3xNd<N>)));
  cudaCheck(cudaMalloc(&hits_geGPU, Rfit::maxNumberOfTracks()*sizeof(Rfit::Matrix6xNf<N>)));
  cudaCheck(cudaMalloc(&fast_fit_resultsGPU, Rfit::maxNumberOfTracks()*sizeof(Rfit::Vector4d)));
  cudaCheck(cudaMalloc(&line_fit_resultsGPU, Rfit::maxNumberOfTracks()*sizeof(Rfit::line_fit)));
  cudaCheck(cudaMalloc(&circle_fit_resultsGPU, Rfit::maxNumberOfTracks()*sizeof(Rfit::circle_fit)));

  cudaCheck(cudaMemset(fast_fit_resultsGPU, 0, Rfit::maxNumberOfTracks()*sizeof(Rfit::Vector4d)));
  cudaCheck(cudaMemset(line_fit_resultsGPU, 0, Rfit::maxNumberOfTracks()*sizeof(Rfit::line_fit)));


  cudaMallocManaged(&gen_parGPU, sizeof(Rfit::Vector6dd));
  gen_parGPU[0] = gen_par;

#ifndef NB
#define NB 1
#endif
#ifndef NT
#define NT 128
#endif

  int ntr = NT;

  int nbl0 = Ntracks/ntr;

  int nbl = NB;


  int64_t * tg;
  cudaMallocManaged(&tg, nbl*sizeof(int64_t));

  kernelPrintSizes<N><<<nbl0, ntr>>>(hitsGPU,hits_geGPU);
  kernelFillHitsAndHitsCov<N><<<nbl0, ntr>>>(gen_parGPU, hitsGPU,hits_geGPU);

  // FAST_FIT GPU
  kernelFastFit<N><<<nbl, ntr>>>(hitsGPU, fast_fit_resultsGPU,Ntracks,tg);
  cudaDeviceSynchronize();

  std::cout << "fastfit gtime ";
  for (int i=0; i<nbl; ++i) std::cout << tg[i] <<  ' ';
  std::cout << '\n' << std::endl;
  cudaDeviceSynchronize();

  
  cudaMemcpy(fast_fit_resultsGPUret, fast_fit_resultsGPU, Rfit::maxNumberOfTracks()*sizeof(Rfit::Vector4d), cudaMemcpyDeviceToHost);
  Rfit::Map4d fast_fit(fast_fit_resultsGPUret+10,4);
  std::cout << "Fitted values (FastFit, [X0, Y0, R, tan(theta)]): GPU\n" << fast_fit << std::endl;
  //assert(isEqualFuzzy(fast_fit_results, fast_fit));
#endif

#ifdef USE_BL
  // CIRCLE AND LINE FIT CPU
  BrokenLine::PreparedBrokenLineData<N> data;
  BrokenLine::karimaki_circle_fit circle_fit_results;
  Rfit::line_fit line_fit_results;
  Rfit::Matrix3d Jacob;
  kk=0;
#ifdef  NOGPU
for (uint32_t k=0; k<32*Ntracks; ++k,kk=k/32) 
#endif
{
  assert(kk<Ntracks);
  BrokenLine::prepareBrokenLineData(hits,fast_fit_results[kk],B,data);
  BrokenLine::BL_Line_fit(hits_ge,fast_fit_results[kk],B,data,line_fit_results);
  BrokenLine::BL_Circle_fit(hits,hits_ge,fast_fit_results[kk],B,data,circle_fit_results);
  Jacob << 1.,0,0,
    0,1.,0,
    0,0,-B/std::copysign(Rfit::sqr(circle_fit_results.par(2)),circle_fit_results.par(2));
  circle_fit_results.par(2)=B/fabs(circle_fit_results.par(2));
  circle_fit_results.cov=Jacob*circle_fit_results.cov*Jacob.transpose();
  /*
  assert(toSingle(line_fit_results.par(0))>=0);
  assert(toSingle(line_fit_results.cov(0,0))>=0);
  assert(toSingle(circle_fit_results.par(2))>=0);
  assert(toSingle(circle_fit_results.cov(0,0))>=0);
  */
}

#ifndef NOGPU
  // fit on GPU
  kernelBrokenLineFit<N><<<nbl, ntr>>>(hitsGPU, hits_geGPU,
					  fast_fit_resultsGPU, B,
					  circle_fit_resultsGPU,
					  line_fit_resultsGPU,
                                          Ntracks,
                                          tg);
  cudaDeviceSynchronize();
  std::cout << "BLfit gtime ";
  for (int i=0; i<nbl; ++i) std::cout << tg[i] <<  ' ';
  std::cout << '\n' << std::endl;
  cudaDeviceSynchronize();
#endif
  
#else
  // CIRCLE_FIT CPU
  Rfit::VectorNd<N> rad = (hits.block(0, 0, 2, N).colwise().norm());

  Rfit::Matrix2Nd<N> hits_cov =  Rfit::Matrix2Nd<N>::Zero();
  Rfit::loadCovariance2D(hits_ge,hits_cov);
  Rfit::circle_fit circle_fit_results = Rfit::Circle_fit(hits.block(0, 0, 2, N),
      hits_cov,
      fast_fit_results, rad, B, true);

  // CIRCLE_FIT GPU
  kernelCircleFit<N><<<Ntracks/64, 64>>>(hitsGPU, hits_geGPU,
      fast_fit_resultsGPU, B, circle_fit_resultsGPU);
  cudaDeviceSynchronize();
 
  // LINE_FIT CPU
  Rfit::line_fit line_fit_results = Rfit::Line_fit(hits, hits_ge, circle_fit_results, fast_fit_results, B, true);


  kernelLineFit<N><<<Ntracks/64, 64>>>(hitsGPU, hits_geGPU, B, circle_fit_resultsGPU, fast_fit_resultsGPU, line_fit_resultsGPU);
  cudaDeviceSynchronize();
#endif

  std::cout << "Fitted values (CircleFit):\n" << circle_fit_results.par << "\nchi2 " << circle_fit_results.chi2 << std::endl;

#ifndef NOGPU  
  cudaMemcpy(circle_fit_resultsGPUret, circle_fit_resultsGPU,
	     sizeof(Rfit::circle_fit), cudaMemcpyDeviceToHost);
  std::cout << "Fitted values (CircleFit) GPU:\n" << circle_fit_resultsGPUret->par << "\nchi2 " << circle_fit_resultsGPUret->chi2 << std::endl;
  // assert(isEqualFuzzy(circle_fit_results.par, circle_fit_resultsGPUret->par));
#endif
  
  std::cout << "Fitted values (LineFit):\n" << line_fit_results.par << "\nchi2 " << line_fit_results.chi2<< std::endl;
#ifndef NOGPU
  // LINE_FIT GPU
  cudaMemcpy(line_fit_resultsGPUret, line_fit_resultsGPU, sizeof(Rfit::line_fit), cudaMemcpyDeviceToHost);
  std::cout << "Fitted values (LineFit) GPU:\n" << line_fit_resultsGPUret->par << "\nchi2 " << line_fit_resultsGPUret->chi2 << std::endl;
//  assert(isEqualFuzzy(line_fit_results.par, line_fit_resultsGPUret->par, N==5 ? 1e-4 : 1e-6)); // requires fma on CPU
#endif
  
  std::cout << "Fitted cov (CircleFit) CPU:\n" << circle_fit_results.cov << std::endl;
  std::cout << "Fitted cov (LineFit): CPU\n" << line_fit_results.cov << std::endl;
#ifndef NOGPU
  std::cout << "Fitted cov (CircleFit) GPU:\n" << circle_fit_resultsGPUret->cov << std::endl;
  std::cout << "Fitted cov (LineFit): GPU\n" << line_fit_resultsGPUret->cov << std::endl;

  cudaCheck(cudaFree(hitsGPU));
  cudaCheck(cudaFree(hits_geGPU));
  cudaCheck(cudaFree(fast_fit_resultsGPU));
  cudaCheck(cudaFree(line_fit_resultsGPU));
  cudaCheck(cudaFree(circle_fit_resultsGPU));
#endif
}

int main (int argc, char * argv[]) {
#ifndef NOGPU
  exitSansCUDADevices();
#endif

  testFit<4>();
//  testFit<3>();
//  testFit<5>();

  std::cout << "TEST FIT, NO ERRORS" << std::endl;

  return 0;
}

