#include <random>
#include <iostream>
#include <iomanip>
#include <ios>

#include <thread>
#include <mutex>

#include "fastNormalPDF.h"

#include "benchmark.h"

#include<algorithm>
int main (int argc, char * argv[]) {
  std::cout << std::setprecision(9); // std::hexfloat;
  std::mt19937_64 gen0;
  std::mt19937_64 gen1;
  std::mt19937_64 gen2;
  std::mt19937_64 gen3;
  std::cout << gen1() << ' ' << gen2()  << ' ' << gen3() << std::endl;
  {
  auto [q,w] = fastNormalPDF::from23(gen1());
  std::cout << q << ' ' << w << std::endl;
  }{
  auto [q,w] = fastNormalPDF::from32(gen2());
  std::cout << q << ' ' << w << std::endl;
  }{
  auto [q,w] = fastNormalPDF::fromMix(gen3());
  std::cout << q << ' ' << w << std::endl;
  }


  {

      auto fgen = [&](uint32_t const * __restrict__ dummy, float *__restrict__ out, int N) {
           fastNormalPDF::genArray(fastNormalPDF::from23,gen1,out,N);
      };

      std::cout << "test 23 bits" << std::endl;
      int N = 10 * 1000 * 1000;
      benchmark::TimeIt bench;
      // fill in batch of 256
      float rv[256];
      uint32_t dummy[1];
      for (int i = 0; i < N; ++i) {
        bench(fgen, dummy, rv, 256);
      }

      std::cout << "duration " << bench.lap() << std::endl;

   }

  {

      auto fgen = [&](uint32_t const * __restrict__ dummy, float *__restrict__ out, int N) {
           fastNormalPDF::genArray(fastNormalPDF::from32,gen1,out,N);
      };

      std::cout << "test 32 bits" << std::endl;
      int N = 10 * 1000 * 1000;
      benchmark::TimeIt bench;
      // fill in batch of 256
      float rv[256];
      uint32_t dummy[1];
      for (int i = 0; i < N; ++i) {
        bench(fgen, dummy, rv, 256);
      }

      std::cout << "duration " << bench.lap() << std::endl;

   }

  {

      auto fgen = [&](uint32_t const * __restrict__ dummy, float *__restrict__ out, int N) {
           fastNormalPDF::genArrayLux(gen0,out,N);
           // fastNormalPDF::genArray(fastNormalPDF::fromMix,gen1,out,N);
      };

      std::cout << "test Mix" << std::endl;
      int N = 10 * 1000 * 1000;
      benchmark::TimeIt bench;
      // fill in batch of 256
      float rv[256];
      uint32_t dummy[1];
      for (int i = 0; i < N; ++i) {
        bench(fgen, dummy, rv, 256);
      }

      std::cout << "duration " << bench.lap() << std::endl;

   }


  benchmark::Histo<200> h1(0.,10.);
  benchmark::Histo<200> h2(0.,10.);
  benchmark::Histo<200> h3(0.,10.);
  float mn[3] = {2.,2.,2.};
  float mx[3] = {-2.,-2.,-2.};
  double av[3]={0,0,0};
  int64_t N=0;
  std::mutex histoLock;

  auto run = [&]() { 
    benchmark::Histo<200> lh1(0.,10.);
    benchmark::Histo<200> lh2(0.,10.);
    benchmark::Histo<200> lh3(0.,10.);  
    int64_t Nl = 1024LL * 1000LL *10LL;
    if (argc>1) Nl *= 100LL;
    float lmn[3] = {2.,2.,2.};
    float lmx[3] = {-2.,-2.,-2.};
    double lav[3]={0,0,0};
    float f1[256];
    float f2[256];
    float f3[256];
    for (int64_t k=0; k<Nl/256; ++k) 
     for (int64_t i=0; i<256; ++i){
      fastNormalPDF::genArray(fastNormalPDF::from23,gen1,f1,256);
      fastNormalPDF::genArray(fastNormalPDF::from32,gen2,f2,256);
      fastNormalPDF::genArray(fastNormalPDF::fromMix,gen3,f3,256);
      lh1(std::abs(f1[i]));
      lh2(std::abs(f2[i]));
      lh3(std::abs(f3[i]));
      lav[0] +=f1[i];
      lav[1] +=f2[i];
      lav[2] +=f3[i];
      lmn[0] = std::min(lmn[0],std::abs(f1[i]));
      lmx[0] = std::max(lmx[0],std::abs(f1[i]));
      lmn[1] = std::min(lmn[1],std::abs(f2[i]));
      lmx[1] = std::max(lmx[1],std::abs(f2[i]));
      lmn[2] = std::min(lmn[2],std::abs(f3[i]));
      lmx[2] = std::max(lmx[2],std::abs(f3[i]));

    }     
    {
     std::lock_guard<std::mutex> guard(histoLock);
     N+=Nl;
     h1.add(lh1);
     h2.add(lh2);
     h3.add(lh3);
     for (int i=0; i<3; ++i) {
      av[i]+=lav[i];
      mn[i] = std::min(lmn[i],mn[i]);
      mx[i] = std::max(lmx[i],mx[i]);
     }
    }

  };


   std::vector<std::thread> th;
   for (int i=0; i<10; i++) th.emplace_back(run);
   for (auto & t:th) t.join();

  auto gauss = [](float x){ return (2.f/std::sqrt(2.f*float(M_PI)))*std::exp(-0.5f*(x*x));};
  std::cout << mn[0] << ' ' << mx[0] << ' ' << av[0]/N << ' ' << h1.chi2(gauss)<< std::endl;
  std::cout << mn[1] << ' ' << mx[1] << ' ' << av[1]/N << ' ' << h2.chi2(gauss) << std::endl;
  std::cout << mn[2] << ' ' << mx[2] << ' ' << av[2]/N << ' ' << h3.chi2(gauss) << std::endl;


  std::cout << std::setprecision(3) << std::scientific;
  std::cout << "from 23" << std::endl;
  h1.printAll(gauss,std::cout);
  std::cout << "from 32" << std::endl;
  h2.printData(std::cout);
  std::cout << "Mix" << std::endl;
  h3.printData(std::cout);
  return 0;
}
