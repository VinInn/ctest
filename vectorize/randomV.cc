#define private public
#include "random"
#include "ext/random"

std::mt19937 eng1;
unsigned int r1[std::mt19937::state_size];
void myrand() {
  for (int i=0; i!=std::mt19937::state_size; ++i) {
    r1[i]=eng1();
  }
}



__gnu_cxx::sfmt11213  seng;
unsigned int s[std::mt19937::state_size];
void sirand() {
  for (int i=0; i!=std::mt19937::state_size; ++i) {
    s[i]=seng();
  }
}



std::mt19937 eng2;
unsigned int r2[std::mt19937::state_size];
void vrand() {
  eng2._M_gen_rand();
  for (int i=0; i!=std::mt19937::state_size; ++i) {
     // Calculate o(x(i)).
      auto __z = eng2._M_x[i];
      __z ^= (__z >> std::mt19937::tempering_u) & std::mt19937::tempering_d;
      __z ^= (__z << std::mt19937::tempering_s) & std::mt19937::tempering_b;
      __z ^= (__z << std::mt19937::tempering_t) & std::mt19937::tempering_c;
      __z ^= (__z >> std::mt19937::tempering_l);
      r2[i] = __z;
   }
}
#undef private

#include<iostream>
#include <x86intrin.h>

unsigned int taux=0;
inline volatile unsigned long long rdtsc() {
 return __rdtscp(&taux);
}


int main() {
   std::cout << std::mt19937::state_size << std::endl;
   long long t1=0, t2=0, t3=0;
   for (int j=0; j!=10000; ++j) {
     t1-=rdtsc();
     myrand();
     t1+=rdtsc();
     t2-=rdtsc();
     vrand();
     t2+=rdtsc();
     t3-=rdtsc();
     sirand();
     t3+=rdtsc();
     for (int i=0; i!=std::mt19937::state_size; ++i) {
      if (r1[i]!=r2[i]) std:: cout << "problem " << j << " " << i << std::endl;
     }
   }
   std:: cout << "times " << t1 << " " << t2 << " " << double(t1)/double(t2) 
	      << " " << t3 << " " << double(t2)/double(t3) << std::endl;
   return 0;
}

