#include <random>
#include <iostream>
#include <iomanip>
#include <ios>


#include "fastNormalPDF.h"

#include "benchmark.h"

#include<algorithm>
int main (int argc, char * argv[]) {
  std::cout << std::setprecision(9); // std::hexfloat;
  std::mt19937_64 gen0;
  std::mt19937_64 gen1;
  std::mt19937_64 gen2;
  std::mt19937_64 gen3;
  std::cout << gen1() << ' ' << gen2() << std::endl;
  {
  auto [q,w] = fastNormalPDF::from23(gen1());
  std::cout << q << ' ' << w << std::endl;
  }{
  auto [q,w] = fastNormalPDF::from32(gen2());
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


  benchmark::Histo<100> h1(0.,10.);
  benchmark::Histo<100> h2(0.,10.);
  benchmark::Histo<100> h3(0.,10.);  
  int64_t N = 1024LL * 1000LL *10LL;
  if (argc>1) N *= 100LL;
  float mn[3] = {2.,2.,2.};
  float mx[3] = {-2.,-2.,-2.};
  double av[3]={0,0,0};
  float f1[256];
  float f2[256];
  float f3[256];
  for (int64_t k=0; k<N/256; ++k) 
   for (int64_t i=0; i<256; ++i){
    fastNormalPDF::genArray(fastNormalPDF::from23,gen1,f1,256);
    fastNormalPDF::genArray(fastNormalPDF::from32,gen2,f2,256);
    h1(std::abs(f1[i]));
    h2(std::abs(f2[i]));
    av[0] +=f1[i];
    av[1] +=f2[i];
    mn[0] = std::min(mn[0],std::abs(f1[i]));
    mx[0] = std::max(mx[0],std::abs(f1[i]));
    mn[1] = std::min(mn[1],std::abs(f2[i]));
    mx[1] = std::max(mx[1],std::abs(f2[i]));
    // mn[2] = std::min(mn[2],f3);
    // mx[2] = std::max(mx[2],f3);
    
  }

  auto gauss = [](float x){ return (2.f/std::sqrt(2.f*float(M_PI)))*std::exp(-0.5f*(x*x));};
  std::cout << mn[0] << ' ' << mx[0] << ' ' << av[0]/N << ' ' << h1.chi2(gauss)<< std::endl;
  std::cout << mn[1] << ' ' << mx[1] << ' ' << av[1]/N << ' ' << h2.chi2(gauss) << std::endl;
  // std::cout << mn[2] << ' ' << mx[2] << ' ' << av[2]/N << ' ' << h3.chi2([](float){return 1.;}) << std::endl;

  h1.printAll(gauss,std::cout);

  return 0;
}
