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
