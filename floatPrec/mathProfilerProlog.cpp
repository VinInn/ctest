// compile with c++ -O2 -fPIC -shared mathProfiler.cpp -o mathProfiler.so -ldl
// run as setenv LD_PRELOAD ./mathProfiler.so ; ./a.out; unsetenv LD_PRELOAD ./mathProfiler.so
// or as  export LD_PRELOAD=./mathProfiler.so; ./a.out; export LD_PRELOAD=
#include <cstdint>
#include <dlfcn.h>
#include <unistd.h>
#include <cstring>

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

  std::string functions[] = {"sincos","atan2+","atan2/","hypot+","hypot/","powx","powy",
  "acos","acosh","asin","asinh","atan","atanh","cbrt","cos","cospi","cosh","erf","erfc","exp","exp10","exp2","expm1","j0","j1","log","log10","log1p","log2","rsqrt","sin","sinpi","sinh","tan","tanpi","tanh","y0","y1","lgamma","tgamma"};

  constexpr int linMax = 32;
  constexpr int logMax = 128;

  struct Stat {
    Stat() : tot(0) {
     for (int i=0; i<256; ++i)  { lin[i]=0; log[i]=0; }
    } 
    std::atomic<uint64_t> lin[256];
    std::atomic<uint64_t> log[256];
    std::atomic<uint64_t> tot;
  };

  std::vector<Stat> stat(2*std::size(functions)); 

  template<typename T>
  void count(T x,int i) {
    stat[i].tot++;
    // fill lin between -max and max
    auto y = std::clamp(x,-T(linMax),T(linMax));
    constexpr T den = 0.5/T(linMax);
    int bin = std::clamp(int(T(256)*den*(y+T(linMax))),0,255);
//    std::cout << ">>> " << i << ' ' << x << ' ' << y << ' ' << bin << std::endl;
    stat[i].lin[bin]++;
    // fill log using just the exponent
    if constexpr (4==sizeof(T)) {
      uint32_t k;
      y = std::abs(x);
      memcpy(&k,&y,4);
      k = k>>23;
      assert(k<256);
      stat[i].log[k]++;
    } else {
      uint64_t k;
      y = std::abs(x);
      memcpy(&k,&y,8);
      k = k>>52;
      assert(k<2048);
      int bin =   k - 1023;
      bin = std::clamp(bin,-127,128) + 127;
      assert(bin<256);
      stat[i].log[bin]++;
    }
  }



  struct Banner {
    Banner(){
      std::cout << "MathProfiler Initialize for " << std::size(functions) << " functions" << std::endl;
      // n.reserve(2*std::size(functions));
      // for ( uint32_t i=0;  i <  2*std::size(functions); i++ ) n[i]=0;
     }

     ~Banner() {
        std::cout  << "MathProfiler finalize " << std::endl;
        int i = 0;
        for ( auto f : functions) {
         std::cout << f+"f_lin " << stat[i].tot << " : ";
         for ( auto const & v : stat[i].lin) std::cout << v << ' ';
         std::cout << std::endl;
         std::cout << f+"f_log " << stat[i].tot << " : ";
         for ( auto const & v : stat[i].log) std::cout << v << ' ';
         std::cout << std::endl;

         std::cout << f+"_lin  " << stat[i+1].tot << " : ";
         for ( auto const & v : stat[i+1].lin) std::cout << v << ' ';
         std::cout << std::endl;
         std::cout << f+"_log  " << stat[i+1].tot << " : ";
         for ( auto const & v : stat[i+1].log) std::cout << v << ' ';
         std::cout << std::endl;
         i+=2;
       }
     }
  };

   Banner banner;

}



//------------------------------------
// manually coded

extern "C"
{
typedef void (*sincosSym)(double, double *, double*);
typedef void (*sincosfSym)(float, float *, float*);


sincosSym origsincos = nullptr;
sincosfSym origsincosf = nullptr;

void sincosf(float x, float *sin, float *cos) {
  if (!origsincosf) origsincosf = (sincosfSym)dlsym(RTLD_NEXT,"sincosf");
  origsincosf(x,sin,cos);
  count(x,0);
}

void sincos(double x, double *sin, double *cos) {
  if (!origsincos) origsincos = (sincosSym)dlsym(RTLD_NEXT,"sincos");
  origsincos(x,sin,cos);
  count(x,1);
}


typedef float (*fun2fSym) (float,float);
typedef double (*fun2dSym) (double,double);

fun2fSym origatan2f = nullptr;
float atan2f(float x, float y) {
  if (!origatan2f) origatan2f = (fun2fSym)dlsym(RTLD_NEXT,"atan2f");
  float ret  = origatan2f(x,y);
  count(x+y, 2 );
  count(x/y, 4 );
  return ret;
}

fun2dSym origatan2d = nullptr;
double atan2(double x, double y) {
  if (!origatan2d) origatan2d = (fun2dSym)dlsym(RTLD_NEXT,"atan2");
  double ret  = origatan2d(x,y);
  count(x+y, 3 );
  count(x/y, 5 );
  return ret;
}


fun2fSym orighypotf = nullptr;
float hypotf(float x, float y) {
  if (!orighypotf) orighypotf = (fun2fSym)dlsym(RTLD_NEXT,"hypotf");
  float ret  = orighypotf(x,y);
  count(x+y, 6 );
  count(x/y, 8 );
  return ret;
}

fun2dSym orighypotd = nullptr;
double hypot(double x, double y) {
  if (!orighypotd) orighypotd = (fun2dSym)dlsym(RTLD_NEXT,"hypot");
  double ret  = orighypotd(x,y);
  count(x+y, 7 );
  count(x/y, 9 );
  return ret;
}


fun2fSym origpowf = nullptr;
float powf(float x, float y) {
  if (!origpowf) origpowf = (fun2fSym)dlsym(RTLD_NEXT,"powf");
  float ret  = origpowf(x,y);
  count(x, 10 );
  count(x, 12);
  return ret;
}

fun2dSym origpowd = nullptr;
double pow(double x, double y) {
  if (!origpowd) origpowd = (fun2dSym)dlsym(RTLD_NEXT,"pow");
  double ret  = origpowd(x,y);
  count(x, 11 );
  count(x, 13);
  return ret;
}

typedef float (*funfSym) (float);
typedef double (*fundSym) (double);

}
