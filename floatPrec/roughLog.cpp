#include <cstdint>
inline float roughLog(float x) {
  // max diff [0.5,12] at 1.25 0.16143
  // average diff  0.0662998
  union IF {uint32_t i; float f;};
  IF z; z.f=x;
  uint32_t lsb = 1<21;
  z.i +=lsb;
  z.i >>= 21;
  auto f = z.i&3;
  int ex = int(z.i >>2) -127;

  // log2(1+0.25*f)
  // averaged over bins
  const float frac[4] = {0.160497f,0.452172f,0.694562f,0.901964f};
  return float(ex)+frac[f];
  
}


#include<iostream>
#include<cmath>

int main() {

     float mdiff = 0;
     float xmax=0;
     double ave = 0;
     double n=0;
     for (auto x=0.5f; x<12.f; x+=0.01f) {
       auto diff = std::abs(std::log2(x)-roughLog(x));
       ave+=diff; n++;
       // std::cout << x << ' ' << diff << std::endl;
       if (diff>mdiff) {
           mdiff = diff;
           xmax = x;
       }
     }

   std::cout << "ave " << ave/n << " max diff at " << xmax << ' ' << mdiff << std::endl;
   std::cout << roughLog(xmax) << ' ' << std::log2(xmax) << std::endl;

   std::cout << std::endl;

  const float frac[4] = {0.f,0.25f,0.5f,0.75f};
  for (auto f: frac) std::cout << std::log2(1.f+f) << ',';
  std::cout << std::endl;
  std::cout << std::endl;
  // take average over bin
  for (auto f: frac) {
   double ave=0, n=0;
   for (float x =0.; x<0.25; x+=0.01) {
     ave+=std::log2(1.f+f+x); n++;
   }
   std::cout<< ave/n <<',';
  }
  std::cout << std::endl;
  std::cout << std::endl;


  std::cout << roughLog(0.5f) << std::endl;
  std::cout << roughLog(0.25f) << std::endl;
  std::cout << roughLog(0.75f) << std::endl;


  std::cout << roughLog(1.0f) << std::endl;
  std::cout << roughLog(4.f) << std::endl;
  std::cout << roughLog(5.f) << std::endl;
  std::cout << roughLog(6.f) << std::endl;
  std::cout << roughLog(7.f) << std::endl;
  std::cout << roughLog(8.f) << std::endl;
  std::cout << roughLog(9.f) << std::endl;



  // asssume chi2 to be c1 at 1 and c2 at Xmax 
  // chi2 = c1 + (c2-c1)/ln(Xmax)*ln(x)
  // so slope = c2-c1)/ln(Xmax)
  constexpr float c1 = 0.9f;
  constexpr float c2 = 1.8;
  constexpr float Xmax = 10.f;
  constexpr float slope =  (c2-c1)/log2(Xmax);
  std::cout << " slope " << slope << std::endl;

  for (auto x=0.5f; x<12.f; x+=0.5f) 
    std::cout << x << ' ' << c1+slope*roughLog(x) << std::endl;


}
