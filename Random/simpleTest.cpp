#include "benchmark.h"

#include "Random64Bits.h"
#include "Xoshiro.h"
#include <random>
#include <ext/random>


#include<string>
#include<cstring>

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
  static_assert(0 == Engine::min());

  static constexpr int numberOfBits = 64 - __builtin_clzll(Engine::max());

  std::cout << "testing engine " << typeid(engine).name() << ' ' << numberOfBits << std::endl;
  int N = 2 * 1000 * 1000;
  int64_t vr[64];
  for (int i = 0; i < 64; ++i)
    vr[i] = 0;
  for (int i = 0; i < N; ++i) {
    auto r =  engine();
    fillBits(vr, r);
  }
  int64_t t = 1000 * 1000;
  for (uint32_t i = 0; i < 64; ++i) {
    if (i>=numberOfBits &&  vr[i]!=0 ) 
      std::cout << "not zero? " << i << ' ' << vr[i] << std::endl;
    if (i<numberOfBits && std::abs(vr[i] - t) > 3000)
      std::cout << "off? " << i << ' ' << vr[i] << std::endl;
  }

  auto gen = [&](uint64_t const*, uint64_t * r, int N) {
    for (int i=0; i<N; ++i) r[i] = engine();
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


#include <typeinfo>
template<typename Engine>
void doTestV(Engine & engine)
{
  
  std::cout << "Testing vector engine "  << typeid(engine).name() << ' ' << Engine::vector_size << std::endl;
  int N = 2 * 1000 * 1000;

  auto gen = [&](uint64_t const*, uint64_t * r, int N) {
    for (int i=0; i<N; i+=Engine::vector_size) {
      auto x =  engine.next();
      std::memcpy(r+i,&x,Engine::vector_size*sizeof(uint64_t));
    }
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

   std::mt19937 stdtw32;
   std::mt19937_64 stdtw64;
   __gnu_cxx::sfmt19937_64 stdtwV64;
   XoshiroSS xoshiross;
   XoshiroPP xoshiropp;
   XoshiroP xoshirop;
   Xoshiro<XoshiroType::TwoSums,uint64_t> xoshirosc;
   XoshiroPP xoshiroppV;

/*
   Random64Bits<std::mt19937>> rbstd32(stdtw32);
   Random64Bits<std::mt19937_64>> rbstd64(stdtw64);
   Random64Bits<__gnu_cxx::sfmt19937_64>>  rbstdV64(stdtwV64);
   Random64Bits<XoshiroSS>> rbXori(xoshiross);
*/

   doTest(stdtw32);
   doTest(stdtw64);
   doTest(stdtwV64);
   doTest(xoshiross);
   doTest(xoshiropp);
   doTest(xoshirop);
   doTest(xoshirosc);
   doTestV(xoshiroppV);

/*
   doTest(rbstd32);
   doTest(rbstd64);
   doTest(rbstdV64);
   doTest(rbXori);
*/
   return 0;
}

