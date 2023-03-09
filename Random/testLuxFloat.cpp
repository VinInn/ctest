#include <random>
#include <iostream>
#include <iomanip>
#include <ios>
#include "canonical.h"
#include "luxFloat.h"

#include "benchmark.h"

#include<algorithm>
int main (int argc, char * argv[]) {
  std::cout << std::setprecision(9); // std::hexfloat;
  constexpr canonical_float_random<float, std::mt19937> canonical_dist;
  std::mt19937 gen0;
  std::mt19937 gen1;
  std::mt19937_64 sgen2; OneIntGen<std::mt19937_64> gen2(sgen2);
  std::mt19937_64 gen3;
  std::cout << gen1() << ' ' << gen2() << std::endl;
  std::cout << canonical_dist (gen1) << std::endl;
  std::cout << luxFloat(gen2) << std::endl;
  // uint64_t r3 = gen3()|(uint64_t(gen3())<<32);
  std::cout << fastFloat<41>(gen3()) << std::endl;


  constexpr float den = 1./(std::numeric_limits<uint32_t>::max()+1.);
  auto fgen0 = [&](uint32_t const*, float * r, int N) {
    for (int i=0; i<N; ++i) r[i] =  den*float(gen0());
  };


  auto fgen1 = [&](uint32_t const*, float * r, int N) {
    for (int i=0; i<N; ++i) r[i] = canonical_dist(gen1);
  };

  auto fgen2 = [&](uint32_t const*, float * r, int N) {
    for (int i=0; i<N; ++i) r[i] = luxFloat(gen2);
  };

  auto fgen3 = [&](uint32_t const*, float * r, int N) {
     uint64_t rr[N];
    for (int i=0; i<N; ++i) 
        rr[i] = gen3();
    for (int i=0; i<N; ++i)
        r[i] = fastFloat<41>(rr[i]);
  };

  {

      std::cout << "test classic" << std::endl;
      int N = 10 * 1000 * 1000;

      benchmark::TimeIt bench;
      // fill in batch of 256
      float rv[256];
      uint32_t dummy[1];
      for (int i = 0; i < N; ++i) {
        bench(fgen0, dummy, rv, 256);
      }

      std::cout << "duration " << bench.lap() << std::endl;

   }

  {

      std::cout << "test web-canonical" << std::endl;
      int N = 10 * 1000 * 1000;

      benchmark::TimeIt bench;
      // fill in batch of 256
      float rv[256];
      uint32_t dummy[1];
      for (int i = 0; i < N; ++i) {
        bench(fgen1, dummy, rv, 256);
      }

      std::cout << "duration " << bench.lap() << std::endl;

   }


  {

      std::cout << "test luxFloat" << std::endl;
      int N = 10 * 1000 * 1000;

      benchmark::TimeIt bench;
      // fill in batch of 256
      float rv[256];
      uint32_t dummy[1];
      for (int i = 0; i < N; ++i) {
        bench(fgen2, dummy, rv, 256);
      }

      std::cout << "duration " << bench.lap() << std::endl;

   }


 {

      std::cout << "test fastFloat" << std::endl;
      int N = 10 * 1000 * 1000;

      benchmark::TimeIt bench;
      // fill in batch of 256
      float rv[256];
      uint32_t dummy[1];
      for (int i = 0; i < N; ++i) {
           bench(fgen3, dummy, rv, 256);
      }

      std::cout << "duration " << bench.lap() << std::endl;

   }

  benchmark::Histo<100> h1(0.,1.);
  benchmark::Histo<100> h2(0.,1.);
  benchmark::Histo<100> h3(0.,1.);  
  int64_t N = 1000LL * 1000LL * 1000LL;
  if (argc>1) N *= 100LL;
  float mn[3] = {2.,2.,2.};
  float mx[3] = {-2.,-2.,-2.};
  double av[3]={0,0,0};
  for (int64_t i=0; i<N; ++i) {
    auto f1 = canonical_dist (gen1);
    auto f2 = luxFloat(gen2);
    auto f3 =  fastFloat<41>(gen3());
    h1(f1);
    h2(f2);
    h3(f3);
    av[0] +=f1;
    av[1] +=f2;
    av[2] +=f3;
    mn[0] = std::min(mn[0],f1);
    mx[0] = std::max(mx[0],f1);
    mn[1] = std::min(mn[1],f2);
    mx[1] = std::max(mx[1],f2);
    mn[2] = std::min(mn[2],f3);
    mx[2] = std::max(mx[2],f3);
  }

  std::cout << mn[0] << ' ' << mx[0] << ' ' << av[0]/N << ' ' << h1.chi2([](float){return 1.;})<< std::endl;
  std::cout << mn[1] << ' ' << mx[1] << ' ' << av[1]/N << ' ' << h2.chi2([](float){return 1.;}) << std::endl;
  std::cout << mn[2] << ' ' << mx[2] << ' ' << av[2]/N << ' ' << h3.chi2([](float){return 1.;}) << std::endl;

  h3.printAll([](float){return 1.;},std::cout);

  return 0;
}
