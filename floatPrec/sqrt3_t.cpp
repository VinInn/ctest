/*
f= pow(y,1/3);
f=exp(log(0.5+y)/3);
I=[-0.25;0.5];
filename="/tmp/polyPow";
print("") > filename;
for deg from 2 to 16 do begin
  p = fpminimax(f, deg,[|0,23...|],I, floating, absolute); 
  display=decimal;
  acc=floor(-log2(sup(supnorm(p, f, I, absolute, 2^(-40)))));
  print( "   // degree = ", deg, 
         "  => absolute accuracy is ",  acc, "bits" ) >> filename;
  print("template<> inline float approx_sqrt3f_P<", deg, ">(float y) {") >> filename;
  display=hexadecimal;
  print("return ", horner(p) , ";") >> filename;
  print("}") >> filename;
end;
*/

template<int DEGREE>
inline float approx_sqrt3f_P(float x);

template<int DEGREE>
inline float approx_sqrt3f(float x) {
  return approx_sqrt3f_P<DEGREE>(x+1.f);
}


   // degree =  2   => absolute accuracy is  2 bits
template<> inline float approx_sqrt3f_P< 2 >(float y) {
return  0x1.p0 + y * (-0x5.ec674p-4 + y * 0x8.eeeb4p-4) ;
}
   // degree =  3   => absolute accuracy is  2 bits
template<> inline float approx_sqrt3f_P< 3 >(float y) {
return  0x1.p0 + y * (-0x1.2faa04p0 + y * (0x3.378e08p0 + y * (-0x1.eb426p0))) ;
}
   // degree =  4   => absolute accuracy is  2 bits
template<> inline float approx_sqrt3f_P< 4 >(float y) {
return  0x1.p0 + y * (-0x2.4d0948p0 + y * (0xa.09fc2p0 + y * (-0xd.c2bc4p0 + y * 0x6.317c5p0))) ;
}
   // degree =  5   => absolute accuracy is  2 bits
template<> inline float approx_sqrt3f_P< 5 >(float y) {
return  0x1.p0 + y * (-0x3.b28f6p0 + y * (0x1.7b4accp4 + y * (-0x3.74b88p4 + y * (0x3.7e1ca8p4 + y * (-0x1.478cd4p4))))) ;
}
   // degree =  6   => absolute accuracy is  2 bits
template<> inline float approx_sqrt3f_P< 6 >(float y) {
return  0x1.p0 + y * (-0x5.5ef3fp0 + y * (0x2.f8ea7p4 + y * (-0xa.5ba6ep4 + y * (0x1.16de5cp8 + y * (-0xe.0b396p4 + y * 0x4.589a6p4))))) ;
}
   // degree =  7   => absolute accuracy is  2 bits
template<> inline float approx_sqrt3f_P< 7 >(float y) {
return  0x1.p0 + y * (-0x7.51d85p0 + y * (0x5.5997ap4 + y * (-0x1.9d4dccp8 + y * (0x3.fb68f8p8 + y * (-0x5.3fefap8 + y * (0x3.8529ep8 + y * (-0xf.17c0ap4))))))) ;
}
   // degree =  8   => absolute accuracy is  2 bits
template<> inline float approx_sqrt3f_P< 8 >(float y) {
return  0x1.p0 + y * (-0x9.8b222p0 + y * (0x8.e808ep4 + y * (-0x3.8ac6a8p8 + y * (0xb.e88bep8 + y * (-0x1.6ac924p12 + y * (0x1.88f6fp12 + y * (-0xe.1aa1p8 + y * 0x3.553238p8))))))) ;
}


#include "approx_exp.h"
#include "approx_log.h"

#include<cstdio>
#include<cstdlib>
#include<iostream>


template<typename STD, typename APPROX>
void accTest(STD stdf, APPROX approx, int degree) {
  using namespace approx_math;
  std::cout << std::endl << "launching  exhaustive test for degree " << degree << std::endl;
  binary32 x,r,ref;
  int maxdiff=0;
  int n127=0;
  int n16393=0;
  x.f=1.e-12; // should be 0 but 
  while(x.f<1.) {
    x.ui32++;
    r.f=approx(x.f);
    ref.f=stdf(x.f); // double-prec one  (no hope with -fno-math-errno)
    int d=abs(r.i32-ref.i32);
    if(d>maxdiff) {
      // std::cout << "new maxdiff for x=" << x.f << " : " << d << std::endl;
      maxdiff=d;
    }
    if (d>127) ++n127;
    if (d>16393) ++n16393;
  }
  std::cout << "maxdiff / diff >127 / diff >16393 " << maxdiff << " / " << n127<< " / " << n16393<< std::endl;
}

int main() {
  for (float y =0.0f; y<1.01f; y=y+0.1f)
    std::cout << y << ": " << std::pow(y,1.f/3.f) 
	    << " " << approx_sqrt3f<2>(y)
	    << " " << approx_sqrt3f<8>(y)
	    << std::endl;

  accTest([](float x){return std::pow(x,1.f/3.f);},approx_sqrt3f<2>,2);
  accTest([](float x){return std::pow(x,1.f/3.f);},approx_sqrt3f<4>,4);
  accTest([](float x){return std::pow(x,1.f/3.f);},approx_sqrt3f<6>,6);
  accTest([](float x){return std::pow(x,1.f/3.f);},approx_sqrt3f<8>,8);
}
