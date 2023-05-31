#include "benchmark.h"

#include "Math/RanluxppEngine.h"
#include "Math/MersenneTwisterEngine.h"
#include "Math/MixMaxEngine.h"
#include "Math/StdEngine.h"
#include "Random64Bits.h"
#include "Xoshiro.h"
#include <random>
#include <ext/random>


#include<string>

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
  static_assert((~0ULL)>>(64-Engine::kNumberOfBits) == Engine::MaxInt());
  static_assert(0 == Engine::MinInt());
  static_assert(64 - Engine::kNumberOfBits ==  __builtin_clzll(Engine::MaxInt()));

  std::cout << "testing engine " << engine.Name() << ' ' << Engine::kNumberOfBits << std::endl;
  int N = 2 * 1000 * 1000;
  int64_t vr[64];
  for (int i = 0; i < 64; ++i)
    vr[i] = 0;
  for (int i = 0; i < N; ++i) {
    auto r =  engine.IntRndm();
    fillBits(vr, r);
  }
  int64_t t = 1000 * 1000;
  for (uint32_t i = 0; i < 64; ++i) {
    if (i>=Engine::kNumberOfBits &&  vr[i]!=0 ) 
      std::cout << "not zero? " << i << ' ' << vr[i] << std::endl;
    if (i<Engine::kNumberOfBits && std::abs(vr[i] - t) > 3000)
      std::cout << "off? " << i << ' ' << vr[i] << std::endl;
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
   ROOT::Math::StdEngine<std::mt19937> stdtw32;
   ROOT::Math::StdEngine<std::mt19937_64> stdtw64;
   ROOT::Math::StdEngine<__gnu_cxx::sfmt19937_64> stdtwV64;
   ROOT::Math::StdEngine<XoshiroSS> xoshiross;
   ROOT::Math::StdEngine<XoshiroPP> xoshiropp;
   ROOT::Math::StdEngine<XoshiroP> xoshirop;

   Random64Bits<ROOT::Math::RanluxppEngine2048> rbLux(lux);
   Random64Bits<ROOT::Math::MixMaxEngine<17,0>> rbmx(mmx17);
   Random64Bits<ROOT::Math::MixMaxEngine<240,0>> rbmx2(mmx240);
   Random64Bits<ROOT::Math::MersenneTwisterEngine> rbtw(mtwist);
   Random64Bits<ROOT::Math::StdEngine<std::mt19937>> rbstd32(stdtw32);
   Random64Bits<ROOT::Math::StdEngine<std::mt19937_64>> rbstd64(stdtw64);
   Random64Bits<ROOT::Math::StdEngine<__gnu_cxx::sfmt19937_64>>  rbstdV64(stdtwV64);
   Random64Bits<ROOT::Math::StdEngine<XoshiroSS>> rbXori(xoshiross);

   doTest(lux);
   doTest(mtwist);
   doTest(mmx17);
   doTest(mmx240);
   doTest(stdtw32);
   doTest(stdtw64);
   doTest(stdtwV64);
   doTest(xoshiross);
   doTest(xoshiropp);
   doTest(xoshirop);

   doTest(rbLux);
   doTest(rbmx);
   doTest(rbmx2);
   doTest(rbtw);
   doTest(rbstd32);
   doTest(rbstd64);
   doTest(rbstdV64);
   doTest(rbXori);

   return 0;
}

