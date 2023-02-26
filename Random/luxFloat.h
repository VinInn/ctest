#pragma once
#include<cstdint>
#include<limits>

template<typename RNG>
float luxFloat(RNG & gen) {
    union FInt {
      float f;
      uint32_t i;
    };

   constexpr uint32_t mask = (1<<23) -1;
   static_assert(0x007FFFFF == mask);
   // we assume gen retuns 32 random bits
   uint32_t r = gen();
   // load mantissa
   FInt fi;
   fi.i = r & mask;
   r >>= 23; // remove used bits (9 left...)
   uint32_t exp = 126 - (r ? __builtin_ctz(r) : 9);
   if (!r) while (exp>=0) {
     r = gen();
     uint32_t n = __builtin_ctz(r);
     exp -= n;
     if (r) break;
   }
   fi.i |= (exp <<23);
   return fi.f;
}
