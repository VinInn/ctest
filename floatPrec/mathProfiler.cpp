#include <cstdint>
#include <dlfcn.h>
#include <unistd.h>

#include<cassert>
#include <unordered_map>
#include<map>
#include<array>
#include<vector>
#include<algorithm>
#include <memory>

#include <atomic>
#include <mutex>
#include <thread>

#include<iostream>

namespace {

  std::atomic<uint64_t> n{0}; 

  template<typename T>
  void count(T x,int ) {
    n++;
  }



  struct Banner {
    Banner(){
      std::cout << "MathProfiler Initialize" << std::endl;
     }

     ~Banner() {
        std::cout  << "MathProfiler finalize " << n << std::endl;
     }
  };

   Banner banner;

}



//------------------------------------

extern "C"
{
typedef void (*sincosSym)(double, double *, double*);
typedef void (*sincosfSym)(float, float *, float*);


sincosSym origsincos = nullptr;
sincosfSym origsincosf = nullptr;

void sincos(double x, double *sin, double *cos) {
  if (!origsincos) origsincos = (sincosSym)dlsym(RTLD_NEXT,"sincos");
  origsincos(x,sin,cos);
  count(x,0);
}

void sincosf(float x, float *sin, float *cos) {
  if (!origsincosf) origsincosf = (sincosfSym)dlsym(RTLD_NEXT,"sincosf");
  origsincosf(x,sin,cos);
  count(x,1);
}

typedef float (*funfSym) (float);
typedef double (*fundSym) (double);
}

//-------------------------------

extern "C" 
{

funfSym origFUNCTf = nullptr;
fundSym origFUNCTd = nullptr;

float FUNCTf(float x) {
  if (!origFUNCTf) origFUNCTf = (funfSym)dlsym(RTLD_NEXT,"FUNCf");
  float ret  = origFUNCTf(x);
  count(x,INDEX);
  return ret;
}


double FUNCT(double x) {
  if (!origFUNCTd) origFUNCTd = (fundSym)dlsym(RTLD_NEXT,"FUNC");
  double ret  = origFUNCTd(x);
  count(x,INDEX);
  return ret;
}


} // "C"
