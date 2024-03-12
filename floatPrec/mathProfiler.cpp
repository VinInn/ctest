// compile with c++ -O2 -fPIC -shared mathProfiler.cpp -o mathProfiler.so -ldl
// or c++ -O2 -fPIC -shared mathProfiler.cpp -o mathProfiler.so -ldl -DNORMALIZE -DTOFILE
// run as setenv LD_PRELOAD ./mathProfiler.so ; ./a.out; unsetenv LD_PRELOAD ./mathProfiler.so
// or as  export LD_PRELOAD=./mathProfiler.so; ./a.out; export LD_PRELOAD=
#include <cstdint>
#include <dlfcn.h>
#include <unistd.h>
#include <cstring>
#include <ctime>

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
#include<fstream>
#include<sstream>

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
    const std::clock_t start;
    Banner():  start(std::clock()) {
      std::cout << "MathProfiler Initialize for " << std::size(functions) << " functions in " << program_invocation_short_name << std::endl;
      // n.reserve(2*std::size(functions));
      // for ( uint32_t i=0;  i <  2*std::size(functions); i++ ) n[i]=0;
     }

     ~Banner() {
        double duration = double(std::clock() - start)/CLOCKS_PER_SEC;
        std::cout  << "MathProfiler finalize after " <<duration << " seconds" << std::endl;
#ifdef NORMALIZE
        double invd = 1./duration;
#else
        double invd = 1;
#endif
        int i = 0;
        std::ostream * pout = &std::cout;
#ifdef TOFILE
        std::ostringstream fname;
        fname <<  "/tmp/MathProfiles/" << program_invocation_short_name << getpid();
        std::ofstream file(fname.str());
        pout = &file;
#endif
        auto & out = *pout;
        for ( auto f : functions) {
         out << f+"f_lin " << invd*stat[i].tot << " : ";
         for ( auto const & v : stat[i].lin) out << invd*v << ' ';
         out << std::endl;
         out << f+"f_log " << invd*stat[i].tot << " : ";
         for ( auto const & v : stat[i].log) out << invd*v << ' ';
         out << std::endl;

         out << f+"_lin  " << invd*stat[i+1].tot << " : ";
         for ( auto const & v : stat[i+1].lin) out << invd*v << ' ';
         out << std::endl;
         out << f+"_log  " << invd*stat[i+1].tot << " : ";
         for ( auto const & v : stat[i+1].log) out << invd*v << ' ';
         out << std::endl;
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
// auto generated
extern "C"
{

funfSym origacosf = nullptr;
float acosf(float x) {
  if (!origacosf) origacosf = (funfSym)dlsym(RTLD_NEXT,"acosf");
  float ret  = origacosf(x);
  count(x, 14 );
  return ret;
}

fundSym origacosd = nullptr;
double acos(double x) {
  if (!origacosd) origacosd = (fundSym)dlsym(RTLD_NEXT,"acos");
  double ret  = origacosd(x);
  count(x, 15 );
  return ret;
}


funfSym origacoshf = nullptr;
float acoshf(float x) {
  if (!origacoshf) origacoshf = (funfSym)dlsym(RTLD_NEXT,"acoshf");
  float ret  = origacoshf(x);
  count(x, 16 );
  return ret;
}

fundSym origacoshd = nullptr;
double acosh(double x) {
  if (!origacoshd) origacoshd = (fundSym)dlsym(RTLD_NEXT,"acosh");
  double ret  = origacoshd(x);
  count(x, 17 );
  return ret;
}


funfSym origasinf = nullptr;
float asinf(float x) {
  if (!origasinf) origasinf = (funfSym)dlsym(RTLD_NEXT,"asinf");
  float ret  = origasinf(x);
  count(x, 18 );
  return ret;
}

fundSym origasind = nullptr;
double asin(double x) {
  if (!origasind) origasind = (fundSym)dlsym(RTLD_NEXT,"asin");
  double ret  = origasind(x);
  count(x, 19 );
  return ret;
}


funfSym origasinhf = nullptr;
float asinhf(float x) {
  if (!origasinhf) origasinhf = (funfSym)dlsym(RTLD_NEXT,"asinhf");
  float ret  = origasinhf(x);
  count(x, 20 );
  return ret;
}

fundSym origasinhd = nullptr;
double asinh(double x) {
  if (!origasinhd) origasinhd = (fundSym)dlsym(RTLD_NEXT,"asinh");
  double ret  = origasinhd(x);
  count(x, 21 );
  return ret;
}


funfSym origatanf = nullptr;
float atanf(float x) {
  if (!origatanf) origatanf = (funfSym)dlsym(RTLD_NEXT,"atanf");
  float ret  = origatanf(x);
  count(x, 22 );
  return ret;
}

fundSym origatand = nullptr;
double atan(double x) {
  if (!origatand) origatand = (fundSym)dlsym(RTLD_NEXT,"atan");
  double ret  = origatand(x);
  count(x, 23 );
  return ret;
}


funfSym origatanhf = nullptr;
float atanhf(float x) {
  if (!origatanhf) origatanhf = (funfSym)dlsym(RTLD_NEXT,"atanhf");
  float ret  = origatanhf(x);
  count(x, 24 );
  return ret;
}

fundSym origatanhd = nullptr;
double atanh(double x) {
  if (!origatanhd) origatanhd = (fundSym)dlsym(RTLD_NEXT,"atanh");
  double ret  = origatanhd(x);
  count(x, 25 );
  return ret;
}


funfSym origcbrtf = nullptr;
float cbrtf(float x) {
  if (!origcbrtf) origcbrtf = (funfSym)dlsym(RTLD_NEXT,"cbrtf");
  float ret  = origcbrtf(x);
  count(x, 26 );
  return ret;
}

fundSym origcbrtd = nullptr;
double cbrt(double x) {
  if (!origcbrtd) origcbrtd = (fundSym)dlsym(RTLD_NEXT,"cbrt");
  double ret  = origcbrtd(x);
  count(x, 27 );
  return ret;
}


funfSym origcosf = nullptr;
float cosf(float x) {
  if (!origcosf) origcosf = (funfSym)dlsym(RTLD_NEXT,"cosf");
  float ret  = origcosf(x);
  count(x, 28 );
  return ret;
}

fundSym origcosd = nullptr;
double cos(double x) {
  if (!origcosd) origcosd = (fundSym)dlsym(RTLD_NEXT,"cos");
  double ret  = origcosd(x);
  count(x, 29 );
  return ret;
}


funfSym origcospif = nullptr;
float cospif(float x) {
  if (!origcospif) origcospif = (funfSym)dlsym(RTLD_NEXT,"cospif");
  float ret  = origcospif(x);
  count(x, 30 );
  return ret;
}

fundSym origcospid = nullptr;
double cospi(double x) {
  if (!origcospid) origcospid = (fundSym)dlsym(RTLD_NEXT,"cospi");
  double ret  = origcospid(x);
  count(x, 31 );
  return ret;
}


funfSym origcoshf = nullptr;
float coshf(float x) {
  if (!origcoshf) origcoshf = (funfSym)dlsym(RTLD_NEXT,"coshf");
  float ret  = origcoshf(x);
  count(x, 32 );
  return ret;
}

fundSym origcoshd = nullptr;
double cosh(double x) {
  if (!origcoshd) origcoshd = (fundSym)dlsym(RTLD_NEXT,"cosh");
  double ret  = origcoshd(x);
  count(x, 33 );
  return ret;
}


funfSym origerff = nullptr;
float erff(float x) {
  if (!origerff) origerff = (funfSym)dlsym(RTLD_NEXT,"erff");
  float ret  = origerff(x);
  count(x, 34 );
  return ret;
}

fundSym origerfd = nullptr;
double erf(double x) {
  if (!origerfd) origerfd = (fundSym)dlsym(RTLD_NEXT,"erf");
  double ret  = origerfd(x);
  count(x, 35 );
  return ret;
}


funfSym origerfcf = nullptr;
float erfcf(float x) {
  if (!origerfcf) origerfcf = (funfSym)dlsym(RTLD_NEXT,"erfcf");
  float ret  = origerfcf(x);
  count(x, 36 );
  return ret;
}

fundSym origerfcd = nullptr;
double erfc(double x) {
  if (!origerfcd) origerfcd = (fundSym)dlsym(RTLD_NEXT,"erfc");
  double ret  = origerfcd(x);
  count(x, 37 );
  return ret;
}


funfSym origexpf = nullptr;
float expf(float x) {
  if (!origexpf) origexpf = (funfSym)dlsym(RTLD_NEXT,"expf");
  float ret  = origexpf(x);
  count(x, 38 );
  return ret;
}

fundSym origexpd = nullptr;
double exp(double x) {
  if (!origexpd) origexpd = (fundSym)dlsym(RTLD_NEXT,"exp");
  double ret  = origexpd(x);
  count(x, 39 );
  return ret;
}


funfSym origexp10f = nullptr;
float exp10f(float x) {
  if (!origexp10f) origexp10f = (funfSym)dlsym(RTLD_NEXT,"exp10f");
  float ret  = origexp10f(x);
  count(x, 40 );
  return ret;
}

fundSym origexp10d = nullptr;
double exp10(double x) {
  if (!origexp10d) origexp10d = (fundSym)dlsym(RTLD_NEXT,"exp10");
  double ret  = origexp10d(x);
  count(x, 41 );
  return ret;
}


funfSym origexp2f = nullptr;
float exp2f(float x) {
  if (!origexp2f) origexp2f = (funfSym)dlsym(RTLD_NEXT,"exp2f");
  float ret  = origexp2f(x);
  count(x, 42 );
  return ret;
}

fundSym origexp2d = nullptr;
double exp2(double x) {
  if (!origexp2d) origexp2d = (fundSym)dlsym(RTLD_NEXT,"exp2");
  double ret  = origexp2d(x);
  count(x, 43 );
  return ret;
}


funfSym origexpm1f = nullptr;
float expm1f(float x) {
  if (!origexpm1f) origexpm1f = (funfSym)dlsym(RTLD_NEXT,"expm1f");
  float ret  = origexpm1f(x);
  count(x, 44 );
  return ret;
}

fundSym origexpm1d = nullptr;
double expm1(double x) {
  if (!origexpm1d) origexpm1d = (fundSym)dlsym(RTLD_NEXT,"expm1");
  double ret  = origexpm1d(x);
  count(x, 45 );
  return ret;
}


funfSym origj0f = nullptr;
float j0f(float x) {
  if (!origj0f) origj0f = (funfSym)dlsym(RTLD_NEXT,"j0f");
  float ret  = origj0f(x);
  count(x, 46 );
  return ret;
}

fundSym origj0d = nullptr;
double j0(double x) {
  if (!origj0d) origj0d = (fundSym)dlsym(RTLD_NEXT,"j0");
  double ret  = origj0d(x);
  count(x, 47 );
  return ret;
}


funfSym origj1f = nullptr;
float j1f(float x) {
  if (!origj1f) origj1f = (funfSym)dlsym(RTLD_NEXT,"j1f");
  float ret  = origj1f(x);
  count(x, 48 );
  return ret;
}

fundSym origj1d = nullptr;
double j1(double x) {
  if (!origj1d) origj1d = (fundSym)dlsym(RTLD_NEXT,"j1");
  double ret  = origj1d(x);
  count(x, 49 );
  return ret;
}


funfSym origlogf = nullptr;
float logf(float x) {
  if (!origlogf) origlogf = (funfSym)dlsym(RTLD_NEXT,"logf");
  float ret  = origlogf(x);
  count(x, 50 );
  return ret;
}

fundSym origlogd = nullptr;
double log(double x) {
  if (!origlogd) origlogd = (fundSym)dlsym(RTLD_NEXT,"log");
  double ret  = origlogd(x);
  count(x, 51 );
  return ret;
}


funfSym origlog10f = nullptr;
float log10f(float x) {
  if (!origlog10f) origlog10f = (funfSym)dlsym(RTLD_NEXT,"log10f");
  float ret  = origlog10f(x);
  count(x, 52 );
  return ret;
}

fundSym origlog10d = nullptr;
double log10(double x) {
  if (!origlog10d) origlog10d = (fundSym)dlsym(RTLD_NEXT,"log10");
  double ret  = origlog10d(x);
  count(x, 53 );
  return ret;
}


funfSym origlog1pf = nullptr;
float log1pf(float x) {
  if (!origlog1pf) origlog1pf = (funfSym)dlsym(RTLD_NEXT,"log1pf");
  float ret  = origlog1pf(x);
  count(x, 54 );
  return ret;
}

fundSym origlog1pd = nullptr;
double log1p(double x) {
  if (!origlog1pd) origlog1pd = (fundSym)dlsym(RTLD_NEXT,"log1p");
  double ret  = origlog1pd(x);
  count(x, 55 );
  return ret;
}


funfSym origlog2f = nullptr;
float log2f(float x) {
  if (!origlog2f) origlog2f = (funfSym)dlsym(RTLD_NEXT,"log2f");
  float ret  = origlog2f(x);
  count(x, 56 );
  return ret;
}

fundSym origlog2d = nullptr;
double log2(double x) {
  if (!origlog2d) origlog2d = (fundSym)dlsym(RTLD_NEXT,"log2");
  double ret  = origlog2d(x);
  count(x, 57 );
  return ret;
}


funfSym origrsqrtf = nullptr;
float rsqrtf(float x) {
  if (!origrsqrtf) origrsqrtf = (funfSym)dlsym(RTLD_NEXT,"rsqrtf");
  float ret  = origrsqrtf(x);
  count(x, 58 );
  return ret;
}

fundSym origrsqrtd = nullptr;
double rsqrt(double x) {
  if (!origrsqrtd) origrsqrtd = (fundSym)dlsym(RTLD_NEXT,"rsqrt");
  double ret  = origrsqrtd(x);
  count(x, 59 );
  return ret;
}


funfSym origsinf = nullptr;
float sinf(float x) {
  if (!origsinf) origsinf = (funfSym)dlsym(RTLD_NEXT,"sinf");
  float ret  = origsinf(x);
  count(x, 60 );
  return ret;
}

fundSym origsind = nullptr;
double sin(double x) {
  if (!origsind) origsind = (fundSym)dlsym(RTLD_NEXT,"sin");
  double ret  = origsind(x);
  count(x, 61 );
  return ret;
}


funfSym origsinpif = nullptr;
float sinpif(float x) {
  if (!origsinpif) origsinpif = (funfSym)dlsym(RTLD_NEXT,"sinpif");
  float ret  = origsinpif(x);
  count(x, 62 );
  return ret;
}

fundSym origsinpid = nullptr;
double sinpi(double x) {
  if (!origsinpid) origsinpid = (fundSym)dlsym(RTLD_NEXT,"sinpi");
  double ret  = origsinpid(x);
  count(x, 63 );
  return ret;
}


funfSym origsinhf = nullptr;
float sinhf(float x) {
  if (!origsinhf) origsinhf = (funfSym)dlsym(RTLD_NEXT,"sinhf");
  float ret  = origsinhf(x);
  count(x, 64 );
  return ret;
}

fundSym origsinhd = nullptr;
double sinh(double x) {
  if (!origsinhd) origsinhd = (fundSym)dlsym(RTLD_NEXT,"sinh");
  double ret  = origsinhd(x);
  count(x, 65 );
  return ret;
}


funfSym origtanf = nullptr;
float tanf(float x) {
  if (!origtanf) origtanf = (funfSym)dlsym(RTLD_NEXT,"tanf");
  float ret  = origtanf(x);
  count(x, 66 );
  return ret;
}

fundSym origtand = nullptr;
double tan(double x) {
  if (!origtand) origtand = (fundSym)dlsym(RTLD_NEXT,"tan");
  double ret  = origtand(x);
  count(x, 67 );
  return ret;
}


funfSym origtanpif = nullptr;
float tanpif(float x) {
  if (!origtanpif) origtanpif = (funfSym)dlsym(RTLD_NEXT,"tanpif");
  float ret  = origtanpif(x);
  count(x, 68 );
  return ret;
}

fundSym origtanpid = nullptr;
double tanpi(double x) {
  if (!origtanpid) origtanpid = (fundSym)dlsym(RTLD_NEXT,"tanpi");
  double ret  = origtanpid(x);
  count(x, 69 );
  return ret;
}


funfSym origtanhf = nullptr;
float tanhf(float x) {
  if (!origtanhf) origtanhf = (funfSym)dlsym(RTLD_NEXT,"tanhf");
  float ret  = origtanhf(x);
  count(x, 70 );
  return ret;
}

fundSym origtanhd = nullptr;
double tanh(double x) {
  if (!origtanhd) origtanhd = (fundSym)dlsym(RTLD_NEXT,"tanh");
  double ret  = origtanhd(x);
  count(x, 71 );
  return ret;
}


funfSym origy0f = nullptr;
float y0f(float x) {
  if (!origy0f) origy0f = (funfSym)dlsym(RTLD_NEXT,"y0f");
  float ret  = origy0f(x);
  count(x, 72 );
  return ret;
}

fundSym origy0d = nullptr;
double y0(double x) {
  if (!origy0d) origy0d = (fundSym)dlsym(RTLD_NEXT,"y0");
  double ret  = origy0d(x);
  count(x, 73 );
  return ret;
}


funfSym origy1f = nullptr;
float y1f(float x) {
  if (!origy1f) origy1f = (funfSym)dlsym(RTLD_NEXT,"y1f");
  float ret  = origy1f(x);
  count(x, 74 );
  return ret;
}

fundSym origy1d = nullptr;
double y1(double x) {
  if (!origy1d) origy1d = (fundSym)dlsym(RTLD_NEXT,"y1");
  double ret  = origy1d(x);
  count(x, 75 );
  return ret;
}


funfSym origlgammaf = nullptr;
float lgammaf(float x) {
  if (!origlgammaf) origlgammaf = (funfSym)dlsym(RTLD_NEXT,"lgammaf");
  float ret  = origlgammaf(x);
  count(x, 76 );
  return ret;
}

fundSym origlgammad = nullptr;
double lgamma(double x) {
  if (!origlgammad) origlgammad = (fundSym)dlsym(RTLD_NEXT,"lgamma");
  double ret  = origlgammad(x);
  count(x, 77 );
  return ret;
}


funfSym origtgammaf = nullptr;
float tgammaf(float x) {
  if (!origtgammaf) origtgammaf = (funfSym)dlsym(RTLD_NEXT,"tgammaf");
  float ret  = origtgammaf(x);
  count(x, 78 );
  return ret;
}

fundSym origtgammad = nullptr;
double tgamma(double x) {
  if (!origtgammad) origtgammad = (fundSym)dlsym(RTLD_NEXT,"tgamma");
  double ret  = origtgammad(x);
  count(x, 79 );
  return ret;
}

} // C
