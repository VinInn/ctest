#include "SimpleSinCos.h"
#include "nativeVector.h"


#include<iostream>


int main() {
   constexpr float pi4 = M_PI/4.;
//   constexpr float pi8 = M_PI/16.;

   for (float x=-M_PI; x<(M_PI+0.01); x+=0.01) {
     float s,c;  simpleSinCos(x,s,c);
     if ( std::abs(s-std::sin(x))>1.e-5  || std::abs(c-std::cos(x))> 1.e-5 ) std::cout << x << ' ' << x/M_PI << ' ' << s << '/' << c << ' ' << std::sin(x) << '/' << std::cos(x) << std::endl;
   }


  float x[1024], y[1024];
  for (int i=0; i<1024; ++i)  x[i] = (M_PI*i)/1024;

  for (int i=0; i<1024; ++i) {
     float c;
     simpleSinCos(x[i],y[i],c);
  }

  float ret=0;
  for (int i=0; i<1024; ++i) ret +=y[i];


  using FVect = nativeVector::FVect;

  FVect a,c,s;
  auto vsize = nativeVector::VSIZE;


  for (unsigned int i=0;  i<nativeVector::VSIZE; ++i)  a[i] = (M_PI*i)/vsize;
  simpleSinCos(a,s,c);
  std::cout << a << ' ' << s << ' ' << c << std::endl;

  return ret>1;
}

