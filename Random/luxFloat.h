#pragma once
#include<cstdint>
#include<limits>
#include<cstring>


#pragma once

#include<string>
#include <cstdint>

template<int N>
struct BitStorage{
  static_assert(N<=8);
  using type = uint64_t;
};

template<>
struct BitStorage<4>{
  using type = uint32_t;
};

template<>
struct BitStorage<3>{
  using type = uint32_t;
};

template<>
struct BitStorage<2>{
  using type = uint16_t;
};
template<>
struct BitStorage<1>{
  using type = uint8_t;
};

template<int N,typename Gen>
class NBitsGen {
public:

   using return_type = typename BitStorage<1+(N-1)/8>::type;

   static constexpr uint32_t NBits = N;
   static constexpr uint32_t Shift = NBits;
   static constexpr int NChunks = 64/Shift;
   static constexpr uint64_t mask = (1ULL<<Shift) -1;

   static_assert(N<=8*sizeof(return_type));

   static constexpr uint64_t min() { return 0;}
   static constexpr uint64_t max() { return mask;}

   static_assert(max()<=std::numeric_limits<return_type>::max());

   NBitsGen(Gen& igen) : engine(igen){};

   return_type operator()() {
     if (0==n) {b = engine(); n=NChunks;}
     return_type ret = b&mask;
     b>>=Shift;
     --n;
     return ret;
   }

private:
   Gen & engine;
   uint64_t b=0; 
   int n=0;
};


template<typename Gen>
class  OneIntGen {
public:
  OneIntGen (Gen& igen) : gen(igen){};

  uint32_t operator()() {
    if (ready) {
       ready = false;
       return u32;
    }
    uint64_t x = gen();
    constexpr uint64_t mask = (1ULL<<32) -1;
    u32 =  x &  mask; 
    ready = true;
    return x >> 32;
  }
private:
  Gen & gen;
  uint32_t u32=0;
  bool ready = false;
};


template<typename RNG>
float luxFloat(RNG & gen) {
    union FInt {
      float f;
      uint32_t i;
    };

   constexpr uint32_t mask = (1<<23) -1;
   // we assume gen retuns 32 random bits
   uint32_t r = gen();
   // load mantissa
   FInt fi;
   fi.i = r & mask;
   r &= ~mask; // remove used bits (9 left...)
   // uint32_t r2 = r | (1<<22);  // slower?
   int32_t exp = 126 - (r ? __builtin_clz(r) : 9);
   // int32_t exp = 126 - __builtin_clz(r2);;
   while (!r) {
     r = gen();
     auto n = __builtin_clz(r);
     exp -= n;
     if (exp<0) { exp=0; break;}
   }
   fi.i |= (exp <<23);
   return fi.f;
}



inline
int32_t clzll(uint64_t x) {
  uint32_t u32[2];
  std::memcpy(u32,&x,8);
  return u32[1] ? __builtin_clz(u32[1]) : 
                    32+ __builtin_clz(u32[0]);
}


inline
constexpr float npower2(uint32_t N) {
   float ret = 0.5f;
   while (N--)  ret*=0.5f;
   return ret;
}

// use the  N leading bits to  build a  random float in [0,1[
template<uint32_t N>
float fastFloat(uint64_t r) {
   static_assert(N>=23);
    union FInt {
      float f;
      uint32_t i;
    };

   constexpr uint32_t kSpare = N-23;
   constexpr uint64_t b_mask = ~((~0ULL)>>kSpare);
   constexpr uint64_t m_mask = (~0ULL)>>(64-23);
   constexpr float fmin = npower2(kSpare);
   constexpr float norm = 1./(1.-fmin);
   FInt fi;
   fi.i = (r>>(64-N)) & m_mask;
   r &= b_mask;
/*
   if constexpr (kSpare<=32) {
     uint32_t s = r>>32;
     int32_t exp = 126 - (s ? __builtin_clz(s) : kSpare);
     fi.i |= (exp <<23);
   } else{
*/
     int32_t exp = 126 - (r ? __builtin_clzll(r) : kSpare);
     fi.i |= (exp <<23);
//   }

   // stretch
   return norm*(fi.f-fmin);
}
