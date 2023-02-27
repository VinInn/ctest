#include <random>
#include <iostream>
#include <iomanip>
#include <ios>
#include "canonical.h"
#include "luxFloat.h"

#include "benchmark.h"

#include<algorithm>
int main () {
  std::cout << std::setprecision(9); // std::hexfloat;
  constexpr canonical_float_random<float, std::mt19937> canonical_dist;
  std::mt19937 gen0;
  std::mt19937 gen1;
  std::mt19937 gen2;
  std::cout << gen1() << ' ' << gen2() << std::endl;
  std::cout << canonical_dist (gen1) << std::endl;
  std::cout << luxFloat(gen2) << std::endl;


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
  
  int64_t N = 1LL * 1000LL * 1000LL * 1000LL;
  float mn[2] = {2.,2,};
  float mx[2] = {-2.,-2.};
  double av[2]={0,0};
  for (int64_t i=0; i<N; ++i) {
    auto f1 = canonical_dist (gen1);
    auto f2 = luxFloat(gen2);
    av[0] +=f1;
    av[1] +=f2;
    mn[0] = std::min(mn[0],f1);
    mx[0] = std::max(mx[0],f1);
    mn[1] = std::min(mn[1],f2);
    mx[1] = std::max(mx[1],f2);
  }

  std::cout << mn[0] << ' ' << mx[0] << ' ' << av[0]/N << std::endl;
  std::cout << mn[1] << ' ' << mx[1] << ' ' << av[1]/N << std::endl;

  return 0;
}
