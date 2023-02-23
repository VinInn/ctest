#include "benchmark.h"

#include "Math/RanluxppEngine.h"
#include "Math/MersenneTwisterEngine.h"
#include "Math/MixMaxEngine.h"

#include<string>

#include <iostream>
#include <cassert>


template<typename Engine>
struct RandomTraits{};

template<int N>
struct RandomTraitsBase {
   struct Bits { uint64_t b=0; int n=0;};

   static constexpr uint32_t NBits = N;
   static constexpr uint32_t Shift = 64-NBits;
   static constexpr int NChunks = NBits/Shift;

   template<typename Engine>
   static uint64_t gen(Bits & b, Engine & engine) {
      constexpr uint64_t mask = (1ULL<<Shift) -1;
      if (0==b.n) {b.b = engine.IntRndm(); b.n=NChunks;}
      uint64_t r = engine.IntRndm();
      r |= (b.b&mask)<<NBits;
      b.b>>=Shift;
      --b.n;
      return r;
   }

   template<typename Engine>
   static void fillBits(uint64_t * r, int n, Engine & engine) {
      constexpr uint64_t mask = (1<<Shift) -1;
      for (int i=0; i<n; ++i) r[i] = engine.IntRndm();
      int i=0; for (int j=0; j<n/NChunks+1; ++j) {
        uint64_t b = engine.IntRndm();
        for (int k=0; k<NChunks; ++k) {
          if (i==n) break;
          r[i++] |= (b&mask)<<NBits;
          b>>=Shift;
        }
      }
      assert(i==n);
   }

};

template<>
struct RandomTraits<ROOT::Math::RanluxppEngine2048> : public RandomTraitsBase<48> {

   using Engine = ROOT::Math::RanluxppEngine2048;
};


template<>
struct RandomTraits<ROOT::Math::MixMaxEngine<17,0>> : public RandomTraitsBase<61> {
  using Engine = ROOT::Math::MixMaxEngine<17,0>;
};




template<typename Engine>
class RandomBits {
public:
  
  using Traits = RandomTraits<Engine>;


   RandomBits(Engine & e) : engine(e){}


  uint64_t IntRndm()  {
    return Traits::gen(bits,engine);
//    if (Size==counter) generate();
//    return buffer[counter++];
  }

  static std::string Name() { return "RandomBits<"+std::string(Engine::Name())+'>';}

private:

  static constexpr int Size = 20;

  void generate() {
    Traits::fillBits(buffer,Size,engine);
    counter=0;
  }

  Engine & engine;
  typename Traits::Bits bits;
  uint64_t buffer[Size];
  int counter = Size;
};


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

   RandomBits<ROOT::Math::RanluxppEngine2048> rbLux(lux);
   RandomBits<ROOT::Math::MixMaxEngine<17,0>> rbmx(mmx17);

   doTest(lux);
   // doTest(mtwist);
   doTest(mmx17);
   doTest(mmx240);

   doTest(rbLux);
   doTest(rbmx);

  return 0;
}
