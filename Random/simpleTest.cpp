#include "benchmark.h"

#include "Math/RanluxppEngine.h"
#include "Math/MersenneTwisterEngine.h"
#include "Math/MixMaxEngine.h"


#include <iostream>
#include <cassert>

inline
void fillBits(int64_t * v, uint64_t x) {
  for (int i = 0; i < 64; ++i) {
    v[i] += x & 1;
    x >>= 1;
  }
}

template<typename Engine>
void doTest(Engine & engine)
{
  std::cout << "testing engine " << engine.Name() << std::endl;
  int N = 2 * 1000 * 1000;
  int64_t vr[64];
  for (int i = 0; i < 64; ++i)
    vr[i] = 0;
  for (int i = 0; i < N; ++i) {
    auto r =  engine.IntRndm();
    fillBits(vr, r);
  }
  int64_t t = 1000 * 1000;
  for (int i = 0; i < 64; ++i) {
    if (std::abs(vr[i] - t) > 3000)
      std::cout << "r " << i << ' ' << vr[i] << std::endl;
  }

  auto gen = [&](uint64_t const*, uint64_t * r, int N) {
    for (int i=0; i<N; ++i) r[i] = engine.IntRndm();
  };

 
      benchmark::TimeIt bench;
      // fill in batch of 256
      uint64_t rv[256];
      uint64_t dummy[1];
      for (int i = 0; i < N; ++i) {
        bench(gen, dummy, rv, 256);
      }

      std::cout << "duration " << bench.lap() << std::endl;


}


int main()
{

   ROOT::Math::RanluxppEngine2048 lux(314159265);
   ROOT::Math::MersenneTwisterEngine mtwist;
   ROOT::Math::MixMaxEngine<17,0> mmx17;
   ROOT::Math::MixMaxEngine<240,0> mmx240;

   doTest(lux);
   // doTest(mtwist);
   doTest(mmx17);
   doTest(mmx240);

  return 0;
}
