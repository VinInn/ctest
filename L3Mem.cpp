#include <random>
#include <x86intrin.h>
#include <iostream>

unsigned int taux=0;
inline volatile unsigned long long rdtsc() {
 return __rdtscp(&taux);
}


int main(int argc, char**) {
  std::mt19937 eng;

  std::uniform_real_distribution<float> rgen(0.,1.);

  constexpr int NN = 1024*1024;

  float r[NN];

  std::cout << sizeof(r) << std::endl;

  for (int i=0;i!=NN;++i)
    r[i]=rgen(eng);
 

  constexpr int KK=1000;

  long long t1=0;

  bool err=false;
  float s[KK];
  for (int ok=0; ok!=KK; ++ok) {
    s[ok]=0;
    t1 -= rdtsc();
    for (int i=0;i!=NN;++i)
      s[ok]+=r[i];
    t1 += rdtsc();
    if (ok>0 && s[ok] != s[ok-1]) err=true;
  }

  std::cout << t1 << std::endl;

  if (err) std::cout << "a mess " << std::endl;


  return 0;

}
