#pragma once

#include<string>
#include <cstdint>


template<int N>
struct RandomTraits {
   struct Bits { uint64_t b=0; int n=0;};

   static constexpr uint32_t NBits = N;
   static constexpr uint32_t Shift = 64-NBits;
   static constexpr int NChunks = NBits/Shift;

   template<typename Engine>
   static uint64_t gen(Bits & b, Engine & engine) {
      static_assert(Engine::kNumberOfBits==NBits);
      constexpr uint64_t mask = (1ULL<<Shift) -1;
      if (0==b.n) {b.b = engine.IntRndm(); b.n=NChunks;}
      uint64_t r = engine.IntRndm();
      r |= (b.b&mask)<<NBits;
      b.b>>=Shift;
      --b.n;
      return r;
   }
};

template<>
struct RandomTraits<32> {
   struct Bits { uint64_t b=0; int n=0;};

   static constexpr uint32_t NBits = 32;
   static constexpr uint32_t Shift = 64-NBits;
   static constexpr int NChunks = NBits/Shift;

   template<typename Engine>
   static uint64_t gen(Bits &, Engine & engine) {
      static_assert(Engine::kNumberOfBits==32);
      uint64_t r = engine.IntRndm();
      r <<=32;
      r |= engine.IntRndm();
      return r;
   }
};


template<>
struct RandomTraits<64> {
   struct Bits { uint64_t b=0; int n=0;};

   static constexpr uint32_t NBits = 64;
   static constexpr uint32_t Shift = 0;
   static constexpr int NChunks = 1;

   template<typename Engine>
   static uint64_t gen(Bits &, Engine & engine) {
      static_assert(Engine::kNumberOfBits==64);
      return  engine.IntRndm();
   }
};


template<typename Engine>
class Random64Bits {
public:

  static constexpr uint32_t kNumberOfBits  = 64;
  static constexpr uint64_t MinInt() { return  std::numeric_limits<uint64_t>::min(); }
  static constexpr uint64_t MaxInt() { return  std::numeric_limits<uint64_t>::max(); }

  using Traits = RandomTraits<Engine::kNumberOfBits>;


  Random64Bits(Engine & e) : engine(e){}


  uint64_t IntRndm()  {
    return Traits::gen(bits,engine);
  }

  static std::string Name() { return "Random64Bits<"+std::string(Engine::Name())+'>';}

private:

  Engine & engine;
  typename Traits::Bits bits;
};


