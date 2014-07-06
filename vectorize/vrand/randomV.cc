#define private public
#include "vrandom.h"
#include "random"
#include "ext/random"

namespace vdt {
  typedef std::mersenne_twister_engine<
    uint32_t,
    32, 624, 397, 31,
    0x9908b0dfUL, 11,
    0xffffffffUL, 7,
    0x9d2c5680UL, 15,
    0xefc60000UL, 18, 1812433253UL> mt19937;

}



std::mt19937 eng0;
unsigned int r0[std::mt19937::state_size];
void stdrand() {
  for (int i=0; i!=std::mt19937::state_size; ++i) {
    r0[i]=eng0();
  }
}



vdt::MersenneTwister eng1;
unsigned int r1[vdt::mt19937::state_size];
void myrand() {
  for (int i=0; i!=vdt::mt19937::state_size; ++i) {
    r1[i]=eng1.one();
  }
}



__gnu_cxx::sfmt11213  seng;
unsigned int s[vdt::mt19937::state_size];
void sirand() {
  for (int i=0; i!=vdt::mt19937::state_size; ++i) {
    s[i]=seng();
  }
}



vdt::mt19937 eng2;
vdt::mt19937 eng3;
vdt::MersenneTwister eng4;
vdt::MersenneTwister eng5;
vdt::MersenneTwister eng6;
std::array<unsigned int,vdt::mt19937::state_size> r2;
void vrand() {
  eng2._M_gen_state();
  for (int i=0; i!=vdt::mt19937::state_size; ++i) {
     // Calculate o(x(i)).
      auto __z = eng2._M_x[i];
      __z ^= (__z >> vdt::mt19937::tempering_u) & vdt::mt19937::tempering_d;
      __z ^= (__z << vdt::mt19937::tempering_s) & vdt::mt19937::tempering_b;
      __z ^= (__z << vdt::mt19937::tempering_t) & vdt::mt19937::tempering_c;
      __z ^= (__z >> vdt::mt19937::tempering_l);
      r2[i] = __z;
   }
}
#undef private

std::array<float,vdt::MersenneTwister::size()> r6;
void frand() {
  eng6.generateState();
  for (int i=0; i!=vdt::MersenneTwister::size(); ++i)
    r6[i]=eng6.fget(i);
}


#include<iostream>
#include<typeinfo>
#include<cassert>
#include <x86intrin.h>

unsigned int taux=0;
inline volatile unsigned long long rdtsc() {
 return __rdtscp(&taux);
}

void verify(unsigned int * __restrict__ r) {
   for (int i=0; i!=vdt::mt19937::state_size; ++i) {
     assert (r1[i]==r[i]);
    }
}

int main() {
  std::cout << vdt::mt19937::state_size << std::endl;
  std::cout << typeid(eng2._M_x[0]).name() << " " << sizeof(eng2._M_x[0]) << std::endl;
  std::cout << typeid(uint_fast32_t).name() << " " << sizeof(uint_fast32_t) << std::endl;

  std::mt19937 engS(0xffff0000);
  vdt::MersenneTwister engM(0xffff0000);

  std::cout << "std " << eng0._M_x[0] << " " <<  eng0._M_x[1] << " " <<  eng0._M_x[623] << std::endl;  
  std::cout << "me  " << eng1.state[0] << " " <<eng1.state[1] << " " << eng1.state[623] << std::endl;  
  std::cout << "std " << engS._M_x[0] << " " <<  engS._M_x[1] << " " <<  engS._M_x[623] << std::endl;  
  std::cout << "me  " << engM.state[0] << " " <<engM.state[1] << " " << engM.state[623] << std::endl;  


  long long t0=0, t1=0, t2=0, t3=0, t4=0, t5=0, t6=0, ts=0;
  stdrand(); myrand(); vrand(); frand();sirand();eng3(r2);eng4(r2.begin(),r2.size()); r2= eng5();
  float l=1000, h=-1000;
  for ( auto f: r6) {
    l = std::min(l,f);
    h = std::max(h,f);
  }
  std::cout << "f min/max " << l << "/" << h << std::endl;

  for (int j=0; j!=100000; ++j) {
    t0-=rdtsc();
    stdrand();
    t0+=rdtsc();
    //
    t1-=rdtsc();
    myrand();
    t1+=rdtsc();
    verify(r1);
    t2-=rdtsc();
    vrand();
    t2+=rdtsc();
    verify(r2.begin());
    t6-=rdtsc();
    frand();
    t6+=rdtsc();
    for ( auto f: r6) { l = std::min(l,f); h = std::max(h,f);}
    t3-=rdtsc();
    eng3(r2);
    t3+=rdtsc();
    verify(r2.begin());
    t4-=rdtsc();
    eng4(r2.begin(),r2.size());
    t4+=rdtsc();
    verify(r2.begin());
    t5-=rdtsc();
    auto r3 = eng5();
    t5+=rdtsc();
    verify(r3.begin());
 
    ts-=rdtsc();
    sirand();
    ts+=rdtsc();
 
  }
  std::cout << "f min/max " << l << "/" << h << std::endl;
  std:: cout << "times   " << t1 << " " << t2 << " " << double(t1)/double(t2) 
	     << "\nstd   " << t0 << " " << double(t1)/double(t0) << " " << double(t2)/double(t0)
	     << "\narr   " << t3 << " " << double(t1)/double(t3) << " " << double(t2)/double(t3)
	     << "\nfloat " << t6 << " " << double(t1)/double(t6) << " " << double(t2)/double(t6)
	     << "\nvect  " << t4 << " " << double(t1)/double(t4) << " " << double(t2)/double(t4)
	     << "\narr   " << t5 << " " << double(t1)/double(t5) << " " << double(t2)/double(t5)
	     << "\nsimd  " << ts << " " << double(t1)/double(ts) << " " << double(t2)/double(ts) << std::endl;
  return 0;
}

