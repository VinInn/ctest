#include<random>
#include<iostream>
#include<fstream>
#include<iomanip>
#include<array>
#include<cmath>
std::mt19937 eng;
std::mt19937 eng2;
std::uniform_real_distribution<double> rgen(0.,1.);

template<typename T>
void put(std::ostream& co, T x) {
  unsigned const char * out = (unsigned const char *)(&x);
  for(int i=0; i<sizeof(T); ++i) co<< out[i];
}


int main() {

  constexpr int N=10000;
  struct P {
    std::array<double,N> phi,eta,r,e;
  };

  P p;

  for(int i=0; i<N; ++i) {
     p.phi[i] = -M_PI +2.*M_PI*rgen(eng);
     p.eta[i] = -5. +10.*rgen(eng);
     p.r[i] =  abs(p.eta[i])<1. ? 100. + 50.*rgen(eng) : 10 + 100*rgen(eng);
     p.e[i] = .1 + 1000.*rgen(eng);
  }
  
  std::ofstream n("native.dat");
  std::ofstream f("floats.dat");
  std::ofstream d32("d32.dat");
  std::ofstream d24("d24.dat");


  unsigned long long m32 = 0xfffff; m32 = ~m32;
  unsigned long long m24 = 0xfffffff;  m24 = ~m24;

  unsigned long long const * lp = (unsigned long long const*)(&p);
  double const * dp = (double const *)(&p);

  auto q32 = lp[0]&m32;
  auto q24 = lp[0]&m24;
  std::cout << std::setprecision(15) << dp[0] << " " << float(dp[0]) << ' ' << *( (double*)(&(lp[0])) ) << ' ' << *( (double*)(&(q32)) )  << ' ' << *( (double*)(&(q24)) ) << std::endl;
  std::cout << std::hexfloat << dp[0] << " " << float(dp[0]) << ' ' << *( (double*)(&(lp[0])) ) << ' ' << *( (double*)(&(q32)) )  << ' ' << *( (double*)(&(q24)) ) << std::endl;

  for(int i=0; i<4*N; ++i) {
    put(n,dp[i]);
    put(f,float(dp[i]));
    put(d32,lp[i]&m32);
    put(d24,lp[i]&m24);
  }

  return 0;

}
