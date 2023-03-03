#include<iostream>
#include<iomanip>
#include<cmath>
#include<limits>
#include<cstdio>
#include<cstring>
#include<bitset>
#include <chrono>
#include "benchmark.h"
#include<array>

#include "simpleSinCos.h"
#include "fastSinCos.h"
#include "sincospi.h"


inline int diff(float x, float y) {
  int i; memcpy(&i,&x,sizeof(int));
  int j; memcpy(&j,&y,sizeof(int));
  return std::abs(i-j);
}

int main() {
  // float ff = 16.f*std::numeric_limits<float>::min();
  float ff = std::numeric_limits<float>::epsilon();
  std::cout << "min " << ff << std::endl;
  int mi; memcpy(&mi,&ff,sizeof(int));
  ff = M_PI;
  int mx; memcpy(&mx,&ff,sizeof(int));
  ff = 1.;
  int mx1; memcpy(&mx1,&ff,sizeof(int));

  ff = M_PI;
  for (float p=-ff; p<=ff; p+=0.1f) {
    std::cout << p << ' ' <<  int ((4./M_PI) * p) << "  " << simpleSin(p) << ' ' << simpleCos(p) << "  "
              << fast_sinf(p) << ' ' << fast_cosf(p) << "  "
              << f32_sinpi(p/ff) << ' ' << f32_cospi(p/ff)<< std::endl;
  }

  {
  int mxDiff=0;
  long long avDiff=0;
  long long n=0;
  float fDiff=0;
  auto loop = [&](int i) {
    float p; memcpy(&p,&i,sizeof(int));
    auto s = float(sin(p));
    auto c = float(cos(p));
    float as = simpleSin(p);
    float ac = simpleCos(p);
    auto rs = std::abs(as-s);
    auto rc = std::abs(ac-c);
    auto sd = diff(s,as);
    auto cd = diff(c,ac);
    avDiff+=sd+cd; n+=2;
    mxDiff=std::max(mxDiff,std::max(sd,cd));
    fDiff=std::max(fDiff,std::max(rs,rc));
  };
  for (auto i=mi; i<=mx; i+=10) loop(i);
  std::cout << n << " Simple diffs " << mxDiff << " " << double(avDiff)/n << std::endl;
  std::cout << fDiff << std::endl;
  }

  {
  int mxDiff=0;
  long long avDiff=0;
  long long n=0;
  float fDiff=0;
  auto loop = [&](int i) {
    float p; memcpy(&p,&i,sizeof(int));
    auto s = float(sin(p));
    auto c = float(cos(p));
    auto as = fast_sinf(p);
    auto ac = fast_cosf(p);
    auto rs = std::abs(as-s);
    auto rc = std::abs(ac-c);
    auto sd = diff(s,as);
    auto cd = diff(c,ac);
    avDiff+=sd+cd; n+=2;
    mxDiff=std::max(mxDiff,std::max(sd,cd));
    fDiff=std::max(fDiff,std::max(rs,rc));
  };
  for (auto i=mi; i<=mx; i+=10) loop(i);
  std::cout << n << " fast diffs " << mxDiff << " " << double(avDiff)/n << std::endl;
  std::cout << fDiff << std::endl;
  }


 {
  int mxDiff=0;
  long long avDiff=0;
  long long n=0;
  float fDiff=0;
  auto loop = [&](int i) {
    float p; memcpy(&p,&i,sizeof(int));
    auto s = float(sin(ff*p));
    auto c = float(cos(ff*p));
    auto as = f32_sinpi(p);
    auto ac = f32_cospi(p);
    auto rs = std::abs(as-s);
    auto rc = std::abs(ac-c);
    auto sd = diff(s,as);
    auto cd = diff(c,ac);
    avDiff+=sd+cd; n+=2;
    mxDiff=std::max(mxDiff,std::max(sd,cd));
    fDiff=std::max(fDiff,std::max(rs,rc));
  };
  for (auto i=mi; i<=mx1; i+=10) loop(i);
  std::cout << n << " f32 diffs " << mxDiff << " " << double(avDiff)/n << std::endl;
  std::cout << fDiff << std::endl;
  }




  
  //timing
  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;

  constexpr int N=1<<8;
  std::cout << "working with batch of " << N << " angles" << std::endl;
  std::array<float,N> p;
  std::array<float,N> x;
  std::array<float,N> y;
  
  auto load = [&](int i, float q) {
     p[i]=q;
    for (int j=1; j<8; ++j)
      p[i+j]=p[i+j-1]+ float(M_PI/4.);
  };
  auto comp = [&](int i) {
    y[i] = simpleSin(p[i]);
    x[i] = simpleCos(p[i]);
    
  };

  auto compf = [&](int i) {
    y[i] = fast_sinf(p[i]);
    x[i] = fast_cosf(p[i]);
    
  };

  auto compf32 = [&](int i) {
    y[i] = f32_sinpi(p[i]);
    x[i] = f32_cospi(p[i]);

  };

  delta = start - start;
  double tot = 0;
  for (auto kk=0; kk<100; ++kk)
  for (float zz=-M_PI; zz< (-M_PI+M_PI/4.-0.001); zz+=4.e-7f) {
    for (auto j=0; j<N; j+=8) {zz+=4.e-7f; load(j,zz); }
    delta -= (std::chrono::high_resolution_clock::now()-start);
    benchmark::touch(p);
    for (auto j=0; j<N; ++j) comp(j);
    benchmark::keep(x);
    benchmark::keep(y);
    delta += (std::chrono::high_resolution_clock::now()-start);
    tot++;
  }

  std::cout <<"Simple Computation took "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()/tot*1000
              << " us" << std::endl;
  double deltaS = std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()/tot;



  delta = start - start;
  delta = start - start;
  tot = 0;
  for (auto kk=0; kk<100; ++kk)
  for (float zz=-M_PI; zz< (-M_PI+M_PI/4.-0.001); zz+=4.e-7f) {
    for (auto j=0; j<N; j+=8) {zz+=4.e-7f; load(j,zz); }
    delta -= (std::chrono::high_resolution_clock::now()-start);
    benchmark::touch(p);
    for (auto j=0; j<N; ++j) compf(j);
    benchmark::keep(x);
    benchmark::keep(y);
    delta += (std::chrono::high_resolution_clock::now()-start);
    tot++;
  }

  std::cout <<"fast Computation took "
	    << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()/tot*1000
              << " us" << std::endl;
  double deltaF = std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()/tot;


  delta = start - start;
  delta = start - start;
  tot = 0;
  for (auto kk=0; kk<100; ++kk)
  for (float zz=-1.; zz< (-1.+1./4.-0.001); zz+=4.e-7f) {
    for (auto j=0; j<N; j+=8) {zz+=4.e-7f; load(j,zz); }
    delta -= (std::chrono::high_resolution_clock::now()-start);
    benchmark::touch(p);
    for (auto j=0; j<N; ++j) compf32(j);
    benchmark::keep(x);
    benchmark::keep(y);
    delta += (std::chrono::high_resolution_clock::now()-start);
    tot++;
  }

  std::cout <<"f32 Computation took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()/tot*1000
              << " us" << std::endl;
  double deltaF32 = std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()/tot;


  std::cout << "f/s " << deltaF/deltaS << std::endl;
  std::cout << "f/f32 " << deltaF/deltaF32 << std::endl;

  return 0;
}
