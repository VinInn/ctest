// compile with c++ -O2 -fPIC -shared mathProfiler.cpp -o mathProfiler.s -ldl
// run as setenv LD_PRELOAD ./mathProfiler.so ; ./a.out; unsetenv LD_PRELOAD ./mathProfiler.so
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

  std::string functions[] = {"sincos","atan2+","atan2/",
  "acos","acosh","asin","asinh","atan","atanh","cbrt","cos","cospi","cosh","erf","erfc","exp","exp10","exp2","expm1","j0","j1","log","log10","log1p","log2","rsqrt","sin","sinpi","sinh","tan","tanpi","tanh","y0","y1","lgamma","tgamma"};

  std::vector<std::atomic<uint64_t>> n(2*std::size(functions)); 

  template<typename T>
  void count(T x,int i) {
    n[i]++;
  }



  struct Banner {
    Banner(){
      std::cout << "MathProfiler Initialize for " << std::size(functions) << " functions" << std::endl;
      // n.reserve(2*std::size(functions));
      for ( uint32_t i=0;  i <  2*std::size(functions); i++ ) n[i]=0;
     }

     ~Banner() {
        std::cout  << "MathProfiler finalize " << std::endl;
        int i = 0;
        for ( auto f : functions) {
         std::cout << f+"f " << n[i] << std::endl;
         std::cout << f+"  " << n[i+1] << std::endl;
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


typedef float (*funfSym) (float);
typedef double (*fundSym) (double);
}

//-------------------------------
// auto generated

extern "C"
{


funfSym origacosf = nullptr;
float acosf(float x) {
  if (!origacosf) origacosf = (funfSym)dlsym(RTLD_NEXT,"acosf");
  float ret  = origacosf(x);
  count(x, 6 );
  return ret;
}

fundSym origacosd = nullptr;
double acos(double x) {
  if (!origacosd) origacosd = (fundSym)dlsym(RTLD_NEXT,"acos");
  double ret  = origacosd(x);
  count(x, 7 );
  return ret;
}


funfSym origacoshf = nullptr;
float acoshf(float x) {
  if (!origacoshf) origacoshf = (funfSym)dlsym(RTLD_NEXT,"acoshf");
  float ret  = origacoshf(x);
  count(x, 8 );
  return ret;
}

fundSym origacoshd = nullptr;
double acosh(double x) {
  if (!origacoshd) origacoshd = (fundSym)dlsym(RTLD_NEXT,"acosh");
  double ret  = origacoshd(x);
  count(x, 9 );
  return ret;
}


funfSym origasinf = nullptr;
float asinf(float x) {
  if (!origasinf) origasinf = (funfSym)dlsym(RTLD_NEXT,"asinf");
  float ret  = origasinf(x);
  count(x, 10 );
  return ret;
}

fundSym origasind = nullptr;
double asin(double x) {
  if (!origasind) origasind = (fundSym)dlsym(RTLD_NEXT,"asin");
  double ret  = origasind(x);
  count(x, 11 );
  return ret;
}


funfSym origasinhf = nullptr;
float asinhf(float x) {
  if (!origasinhf) origasinhf = (funfSym)dlsym(RTLD_NEXT,"asinhf");
  float ret  = origasinhf(x);
  count(x, 12 );
  return ret;
}

fundSym origasinhd = nullptr;
double asinh(double x) {
  if (!origasinhd) origasinhd = (fundSym)dlsym(RTLD_NEXT,"asinh");
  double ret  = origasinhd(x);
  count(x, 13 );
  return ret;
}


funfSym origatanf = nullptr;
float atanf(float x) {
  if (!origatanf) origatanf = (funfSym)dlsym(RTLD_NEXT,"atanf");
  float ret  = origatanf(x);
  count(x, 14 );
  return ret;
}

fundSym origatand = nullptr;
double atan(double x) {
  if (!origatand) origatand = (fundSym)dlsym(RTLD_NEXT,"atan");
  double ret  = origatand(x);
  count(x, 15 );
  return ret;
}


funfSym origatanhf = nullptr;
float atanhf(float x) {
  if (!origatanhf) origatanhf = (funfSym)dlsym(RTLD_NEXT,"atanhf");
  float ret  = origatanhf(x);
  count(x, 16 );
  return ret;
}

fundSym origatanhd = nullptr;
double atanh(double x) {
  if (!origatanhd) origatanhd = (fundSym)dlsym(RTLD_NEXT,"atanh");
  double ret  = origatanhd(x);
  count(x, 17 );
  return ret;
}


funfSym origcbrtf = nullptr;
float cbrtf(float x) {
  if (!origcbrtf) origcbrtf = (funfSym)dlsym(RTLD_NEXT,"cbrtf");
  float ret  = origcbrtf(x);
  count(x, 18 );
  return ret;
}

fundSym origcbrtd = nullptr;
double cbrt(double x) {
  if (!origcbrtd) origcbrtd = (fundSym)dlsym(RTLD_NEXT,"cbrt");
  double ret  = origcbrtd(x);
  count(x, 19 );
  return ret;
}


funfSym origcosf = nullptr;
float cosf(float x) {
  if (!origcosf) origcosf = (funfSym)dlsym(RTLD_NEXT,"cosf");
  float ret  = origcosf(x);
  count(x, 20 );
  return ret;
}

fundSym origcosd = nullptr;
double cos(double x) {
  if (!origcosd) origcosd = (fundSym)dlsym(RTLD_NEXT,"cos");
  double ret  = origcosd(x);
  count(x, 21 );
  return ret;
}


funfSym origcospif = nullptr;
float cospif(float x) {
  if (!origcospif) origcospif = (funfSym)dlsym(RTLD_NEXT,"cospif");
  float ret  = origcospif(x);
  count(x, 22 );
  return ret;
}

fundSym origcospid = nullptr;
double cospi(double x) {
  if (!origcospid) origcospid = (fundSym)dlsym(RTLD_NEXT,"cospi");
  double ret  = origcospid(x);
  count(x, 23 );
  return ret;
}


funfSym origcoshf = nullptr;
float coshf(float x) {
  if (!origcoshf) origcoshf = (funfSym)dlsym(RTLD_NEXT,"coshf");
  float ret  = origcoshf(x);
  count(x, 24 );
  return ret;
}

fundSym origcoshd = nullptr;
double cosh(double x) {
  if (!origcoshd) origcoshd = (fundSym)dlsym(RTLD_NEXT,"cosh");
  double ret  = origcoshd(x);
  count(x, 25 );
  return ret;
}


funfSym origerff = nullptr;
float erff(float x) {
  if (!origerff) origerff = (funfSym)dlsym(RTLD_NEXT,"erff");
  float ret  = origerff(x);
  count(x, 26 );
  return ret;
}

fundSym origerfd = nullptr;
double erf(double x) {
  if (!origerfd) origerfd = (fundSym)dlsym(RTLD_NEXT,"erf");
  double ret  = origerfd(x);
  count(x, 27 );
  return ret;
}


funfSym origerfcf = nullptr;
float erfcf(float x) {
  if (!origerfcf) origerfcf = (funfSym)dlsym(RTLD_NEXT,"erfcf");
  float ret  = origerfcf(x);
  count(x, 28 );
  return ret;
}

fundSym origerfcd = nullptr;
double erfc(double x) {
  if (!origerfcd) origerfcd = (fundSym)dlsym(RTLD_NEXT,"erfc");
  double ret  = origerfcd(x);
  count(x, 29 );
  return ret;
}


funfSym origexpf = nullptr;
float expf(float x) {
  if (!origexpf) origexpf = (funfSym)dlsym(RTLD_NEXT,"expf");
  float ret  = origexpf(x);
  count(x, 30 );
  return ret;
}

fundSym origexpd = nullptr;
double exp(double x) {
  if (!origexpd) origexpd = (fundSym)dlsym(RTLD_NEXT,"exp");
  double ret  = origexpd(x);
  count(x, 31 );
  return ret;
}


funfSym origexp10f = nullptr;
float exp10f(float x) {
  if (!origexp10f) origexp10f = (funfSym)dlsym(RTLD_NEXT,"exp10f");
  float ret  = origexp10f(x);
  count(x, 32 );
  return ret;
}

fundSym origexp10d = nullptr;
double exp10(double x) {
  if (!origexp10d) origexp10d = (fundSym)dlsym(RTLD_NEXT,"exp10");
  double ret  = origexp10d(x);
  count(x, 33 );
  return ret;
}


funfSym origexp2f = nullptr;
float exp2f(float x) {
  if (!origexp2f) origexp2f = (funfSym)dlsym(RTLD_NEXT,"exp2f");
  float ret  = origexp2f(x);
  count(x, 34 );
  return ret;
}

fundSym origexp2d = nullptr;
double exp2(double x) {
  if (!origexp2d) origexp2d = (fundSym)dlsym(RTLD_NEXT,"exp2");
  double ret  = origexp2d(x);
  count(x, 35 );
  return ret;
}


funfSym origexpm1f = nullptr;
float expm1f(float x) {
  if (!origexpm1f) origexpm1f = (funfSym)dlsym(RTLD_NEXT,"expm1f");
  float ret  = origexpm1f(x);
  count(x, 36 );
  return ret;
}

fundSym origexpm1d = nullptr;
double expm1(double x) {
  if (!origexpm1d) origexpm1d = (fundSym)dlsym(RTLD_NEXT,"expm1");
  double ret  = origexpm1d(x);
  count(x, 37 );
  return ret;
}


funfSym origj0f = nullptr;
float j0f(float x) {
  if (!origj0f) origj0f = (funfSym)dlsym(RTLD_NEXT,"j0f");
  float ret  = origj0f(x);
  count(x, 38 );
  return ret;
}

fundSym origj0d = nullptr;
double j0(double x) {
  if (!origj0d) origj0d = (fundSym)dlsym(RTLD_NEXT,"j0");
  double ret  = origj0d(x);
  count(x, 39 );
  return ret;
}


funfSym origj1f = nullptr;
float j1f(float x) {
  if (!origj1f) origj1f = (funfSym)dlsym(RTLD_NEXT,"j1f");
  float ret  = origj1f(x);
  count(x, 40 );
  return ret;
}

fundSym origj1d = nullptr;
double j1(double x) {
  if (!origj1d) origj1d = (fundSym)dlsym(RTLD_NEXT,"j1");
  double ret  = origj1d(x);
  count(x, 41 );
  return ret;
}


funfSym origlogf = nullptr;
float logf(float x) {
  if (!origlogf) origlogf = (funfSym)dlsym(RTLD_NEXT,"logf");
  float ret  = origlogf(x);
  count(x, 42 );
  return ret;
}

fundSym origlogd = nullptr;
double log(double x) {
  if (!origlogd) origlogd = (fundSym)dlsym(RTLD_NEXT,"log");
  double ret  = origlogd(x);
  count(x, 43 );
  return ret;
}


funfSym origlog10f = nullptr;
float log10f(float x) {
  if (!origlog10f) origlog10f = (funfSym)dlsym(RTLD_NEXT,"log10f");
  float ret  = origlog10f(x);
  count(x, 44 );
  return ret;
}

fundSym origlog10d = nullptr;
double log10(double x) {
  if (!origlog10d) origlog10d = (fundSym)dlsym(RTLD_NEXT,"log10");
  double ret  = origlog10d(x);
  count(x, 45 );
  return ret;
}


funfSym origlog1pf = nullptr;
float log1pf(float x) {
  if (!origlog1pf) origlog1pf = (funfSym)dlsym(RTLD_NEXT,"log1pf");
  float ret  = origlog1pf(x);
  count(x, 46 );
  return ret;
}

fundSym origlog1pd = nullptr;
double log1p(double x) {
  if (!origlog1pd) origlog1pd = (fundSym)dlsym(RTLD_NEXT,"log1p");
  double ret  = origlog1pd(x);
  count(x, 47 );
  return ret;
}


funfSym origlog2f = nullptr;
float log2f(float x) {
  if (!origlog2f) origlog2f = (funfSym)dlsym(RTLD_NEXT,"log2f");
  float ret  = origlog2f(x);
  count(x, 48 );
  return ret;
}

fundSym origlog2d = nullptr;
double log2(double x) {
  if (!origlog2d) origlog2d = (fundSym)dlsym(RTLD_NEXT,"log2");
  double ret  = origlog2d(x);
  count(x, 49 );
  return ret;
}


funfSym origrsqrtf = nullptr;
float rsqrtf(float x) {
  if (!origrsqrtf) origrsqrtf = (funfSym)dlsym(RTLD_NEXT,"rsqrtf");
  float ret  = origrsqrtf(x);
  count(x, 50 );
  return ret;
}

fundSym origrsqrtd = nullptr;
double rsqrt(double x) {
  if (!origrsqrtd) origrsqrtd = (fundSym)dlsym(RTLD_NEXT,"rsqrt");
  double ret  = origrsqrtd(x);
  count(x, 51 );
  return ret;
}


funfSym origsinf = nullptr;
float sinf(float x) {
  if (!origsinf) origsinf = (funfSym)dlsym(RTLD_NEXT,"sinf");
  float ret  = origsinf(x);
  count(x, 52 );
  return ret;
}

fundSym origsind = nullptr;
double sin(double x) {
  if (!origsind) origsind = (fundSym)dlsym(RTLD_NEXT,"sin");
  double ret  = origsind(x);
  count(x, 53 );
  return ret;
}


funfSym origsinpif = nullptr;
float sinpif(float x) {
  if (!origsinpif) origsinpif = (funfSym)dlsym(RTLD_NEXT,"sinpif");
  float ret  = origsinpif(x);
  count(x, 54 );
  return ret;
}

fundSym origsinpid = nullptr;
double sinpi(double x) {
  if (!origsinpid) origsinpid = (fundSym)dlsym(RTLD_NEXT,"sinpi");
  double ret  = origsinpid(x);
  count(x, 55 );
  return ret;
}


funfSym origsinhf = nullptr;
float sinhf(float x) {
  if (!origsinhf) origsinhf = (funfSym)dlsym(RTLD_NEXT,"sinhf");
  float ret  = origsinhf(x);
  count(x, 56 );
  return ret;
}

fundSym origsinhd = nullptr;
double sinh(double x) {
  if (!origsinhd) origsinhd = (fundSym)dlsym(RTLD_NEXT,"sinh");
  double ret  = origsinhd(x);
  count(x, 57 );
  return ret;
}


funfSym origtanf = nullptr;
float tanf(float x) {
  if (!origtanf) origtanf = (funfSym)dlsym(RTLD_NEXT,"tanf");
  float ret  = origtanf(x);
  count(x, 58 );
  return ret;
}

fundSym origtand = nullptr;
double tan(double x) {
  if (!origtand) origtand = (fundSym)dlsym(RTLD_NEXT,"tan");
  double ret  = origtand(x);
  count(x, 59 );
  return ret;
}


funfSym origtanpif = nullptr;
float tanpif(float x) {
  if (!origtanpif) origtanpif = (funfSym)dlsym(RTLD_NEXT,"tanpif");
  float ret  = origtanpif(x);
  count(x, 60 );
  return ret;
}

fundSym origtanpid = nullptr;
double tanpi(double x) {
  if (!origtanpid) origtanpid = (fundSym)dlsym(RTLD_NEXT,"tanpi");
  double ret  = origtanpid(x);
  count(x, 61 );
  return ret;
}


funfSym origtanhf = nullptr;
float tanhf(float x) {
  if (!origtanhf) origtanhf = (funfSym)dlsym(RTLD_NEXT,"tanhf");
  float ret  = origtanhf(x);
  count(x, 62 );
  return ret;
}

fundSym origtanhd = nullptr;
double tanh(double x) {
  if (!origtanhd) origtanhd = (fundSym)dlsym(RTLD_NEXT,"tanh");
  double ret  = origtanhd(x);
  count(x, 63 );
  return ret;
}


funfSym origy0f = nullptr;
float y0f(float x) {
  if (!origy0f) origy0f = (funfSym)dlsym(RTLD_NEXT,"y0f");
  float ret  = origy0f(x);
  count(x, 64 );
  return ret;
}

fundSym origy0d = nullptr;
double y0(double x) {
  if (!origy0d) origy0d = (fundSym)dlsym(RTLD_NEXT,"y0");
  double ret  = origy0d(x);
  count(x, 65 );
  return ret;
}


funfSym origy1f = nullptr;
float y1f(float x) {
  if (!origy1f) origy1f = (funfSym)dlsym(RTLD_NEXT,"y1f");
  float ret  = origy1f(x);
  count(x, 66 );
  return ret;
}

fundSym origy1d = nullptr;
double y1(double x) {
  if (!origy1d) origy1d = (fundSym)dlsym(RTLD_NEXT,"y1");
  double ret  = origy1d(x);
  count(x, 67 );
  return ret;
}


funfSym origlgammaf = nullptr;
float lgammaf(float x) {
  if (!origlgammaf) origlgammaf = (funfSym)dlsym(RTLD_NEXT,"lgammaf");
  float ret  = origlgammaf(x);
  count(x, 68 );
  return ret;
}

fundSym origlgammad = nullptr;
double lgamma(double x) {
  if (!origlgammad) origlgammad = (fundSym)dlsym(RTLD_NEXT,"lgamma");
  double ret  = origlgammad(x);
  count(x, 69 );
  return ret;
}


funfSym origtgammaf = nullptr;
float tgammaf(float x) {
  if (!origtgammaf) origtgammaf = (funfSym)dlsym(RTLD_NEXT,"tgammaf");
  float ret  = origtgammaf(x);
  count(x, 70 );
  return ret;
}

fundSym origtgammad = nullptr;
double tgamma(double x) {
  if (!origtgammad) origtgammad = (fundSym)dlsym(RTLD_NEXT,"tgamma");
  double ret  = origtgammad(x);
  count(x, 71 );
  return ret;
}

} // C
