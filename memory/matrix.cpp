#include<iostream>
#include<cstdint>
#include<memory>
#include<algorithm>
#include<chrono>

#include <vector>
#include <array>

// #include "memory_usage.h"

auto start = std::chrono::high_resolution_clock::now();


int main() {

  auto delta1 = start - start;
  auto delta2 = delta1;
  auto delta3 = delta1;
  auto delta4 = delta1;

  constexpr uint32_t M=6, N=10000;
#ifdef USE_CPP
  std::cout << "using c++ vector and arrays" << std::endl;
  std::array<std::vector<float>,M> a{std::vector<float>(N),std::vector<float>(N),std::vector<float>(N),std::vector<float>(N),std::vector<float>(N),std::vector<float>(N)};
  std::vector<std::array<float,M>> b(N);
 
  std::array<float,M> v;
  std::vector<float> w(N); 
 
#else
  std::cout << "using C arrays" << std::endl;
  float a[M][N];
  float b[N][M];

  float v[M];
  float w[N];
#endif
  std::cout << "a["<<M<<"]["<<N<<"] " << &a[0][0] - &a[0][1] << ' ' << &a[0][0] - &a[1][0] << std::endl;
  std::cout << "b["<<N<<"]["<<M<<"] " << &b[0][0] - &b[0][1] << ' ' << &b[0][0] - &b[1][0] << std::endl;



  // init 
  for (uint32_t j=0; j<N; ++j)
    for (uint32_t i=0; i<M; ++i) a[i][j]=b[j][i]= i*10+0.0001*j;
  for (uint32_t i=0; i<M; ++i) v[i]=i+1.;
  for (uint32_t i=0; i<N; ++i) w[i]=0.00001*(i+1);

  
  auto loopV = [&]() {
   for (uint32_t i=0; i<M; ++i) v[i] = 1.f/v[i];

   // heat up 
   float f=0.5f;
   for (uint32_t j=0; j<N; ++j)
    for (uint32_t i=0; i<M; ++i) { f*=-1.f; a[i][j] += f*b[j][i]; }


   delta1 -= (std::chrono::high_resolution_clock::now() -start);
   for (uint32_t j=0; j<N; ++j)
    for (uint32_t i=0; i<M; ++i) a[i][j] *= v[i]; 
   delta1 += (std::chrono::high_resolution_clock::now() -start);


   delta2 -= (std::chrono::high_resolution_clock::now() -start);
   for (uint32_t i=0; i<M; ++i)
        for (uint32_t j=0; j<N; ++j) a[i][j] *= v[i];
   delta2 += (std::chrono::high_resolution_clock::now() -start);


   delta3 -= (std::chrono::high_resolution_clock::now() -start);
   for (uint32_t j=0; j<N; ++j)
    for (uint32_t i=0; i<M; ++i) b[j][i] *= v[i];
   delta3 += (std::chrono::high_resolution_clock::now() -start);

   delta4 -= (std::chrono::high_resolution_clock::now() -start);
   for (uint32_t i=0; i<M; ++i)
        for (uint32_t j=0; j<N; ++j) b[j][i] *= v[i];
   delta4 += (std::chrono::high_resolution_clock::now() -start);

  };


  auto loopW = [&]() {
   for (uint32_t i=0; i<N; ++i) w[i]= 1./w[i];

   delta1 -= (std::chrono::high_resolution_clock::now() -start);
   for (uint32_t j=0; j<N; ++j)
    for (uint32_t i=0; i<M; ++i) a[i][j] *= w[j];
   delta1 += (std::chrono::high_resolution_clock::now() -start);


   delta2 -= (std::chrono::high_resolution_clock::now() -start);
   for (uint32_t i=0; i<M; ++i)
        for (uint32_t j=0; j<N; ++j) a[i][j] *= w[j];
   delta2 += (std::chrono::high_resolution_clock::now() -start);


   delta3 -= (std::chrono::high_resolution_clock::now() -start);
   for (uint32_t j=0; j<N; ++j)
    for (uint32_t i=0; i<M; ++i) b[j][i] *= w[j];
   delta3 += (std::chrono::high_resolution_clock::now() -start);

   delta4 -= (std::chrono::high_resolution_clock::now() -start);
   for (uint32_t i=0; i<M; ++i)
        for (uint32_t j=0; j<N; ++j) b[j][i] *= w[j];
   delta4 += (std::chrono::high_resolution_clock::now() -start);

  };


  constexpr int Niter=10000;
  for (int iter=0; iter<Niter; ++iter) loopV();


  // use results

  float sum=0;
  float f=1;
  for (uint32_t j=0; j<N; ++j) { 
    for (uint32_t i=0; i<M; ++i) { f*=-1; sum += f*(a[i][j]-b[j][i]); }
  }

  std::cout << sum << std::endl;


  double DNNK = Niter*1000;
  std::cout <<" computation took "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(delta1).count()/DNNK << ' '
        << std::chrono::duration_cast<std::chrono::nanoseconds>(delta2).count()/DNNK  << ' '
        << std::chrono::duration_cast<std::chrono::nanoseconds>(delta3).count()/DNNK  << ' '
        << std::chrono::duration_cast<std::chrono::nanoseconds>(delta4).count()/DNNK  << ' '
        << " ms\n" << std::endl;

  delta1=delta2=delta3=delta4=start-start;


 for (int iter=0; iter<Niter; ++iter) loopW();
 // use results

  for (uint32_t j=0; j<N; ++j)
    for (uint32_t i=0; i<M; ++i) sum +=a[i][j]-b[j][i];


  std::cout <<" computation took "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(delta1).count()/DNNK << ' '
        << std::chrono::duration_cast<std::chrono::nanoseconds>(delta2).count()/DNNK  << ' '
        << std::chrono::duration_cast<std::chrono::nanoseconds>(delta3).count()/DNNK  << ' '
        << std::chrono::duration_cast<std::chrono::nanoseconds>(delta4).count()/DNNK  << ' '
        << " ms\n" << std::endl;



  std::cout << sum << std::endl;

  return sum;

}
