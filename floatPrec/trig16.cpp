#include "trig16.h"


#include<iostream>
#include<cmath>
#include<algorithm>


template<typename F>
void error(F sc) {  
   float emaxS = 0; int lS=0;
   float emaxC = 0; int lC=0;
   for (int i = -trig16::Sin15::NBins; i<trig16::Sin15::NBins; ++i) {
     float x = trig16::tof(i);
     auto [s,c] = sc(i);
     auto dS = std::abs(s-std::sin(x));
     auto dC = std::abs(c-std::cos(x));
     if (dS>emaxS) { emaxS=dS; lS=i;}
     if (dC>emaxC) { emaxC=dC; lC=i;}
   }
   std::cout << "emax " << trig16::tof(lS) <<':'<< lS <<':'<< emaxS
            << " , "  <<   trig16::tof(lC) <<':' << lC <<':'<< emaxC << std::endl;
}

int main() {

   using trig16::Sin14;
   std::cout << trig16::pi4 << ' ' << trig16::mask << ' ' << Sin14::coeff << ' ' << Sin14::coefi << std::endl;
   std::cout << Sin14::tof(Sin14::NBins) << ' ' << Sin14::toi(trig16::pi4) << std::endl;
   std::cout << Sin14::tof(-Sin14::NBins) << ' ' << Sin14::toi(-trig16::pi4) << std::endl;


   trig16::Sin14 sin14(std::sin<float>);
   std::cout << sin14(0) << ' ' << sin14(trig16::Sin14::NBins-1) << std::endl;
   std::cout << sin14(1) << ' ' << sin14(trig16::Sin14::NBins-2) << std::endl;

   std::cout << "\n____________\n"  << std::endl;


   double a[4] = {0.5,2.,-2.,-0.5};
   for ( auto x : a ) {
     auto [s,c] = trig16::sincos14(trig16::to16(x));
     std::cout << x << ' ' << trig16::to16(x) << ' ' << std::sin(x)<<',' << std::cos(x)
             << ' ' << s <<',' << c << std::endl;
   }

   std::cout << "\n____________\n"  << std::endl;

    
   std::cout << "sincos14" << std::endl;
   error(trig16::sincos14);
   std::cout << std::endl;

   std::cout << "sincos9" << std::endl;
   error(trig16::sincos9);
   std::cout << std::endl;


   std::cout << "\n____________\n"  << std::endl;

   for ( auto x : a ) {
     auto [s,c] = trig16::sincos9(trig16::to16(x));
     std::cout << x << ' ' << trig16::to16(x) << ' ' << std::sin(x)<<',' << std::cos(x)
             << ' ' << s <<',' << c << std::endl;
   }

   std::cout << "\n____________\n"  << std::endl;

    
   std::cout << trig16::tof(16)/trig16::pi << ' ' << trig16::tof(256)/trig16::pi << ' ' 
             << trig16::tof(512)/trig16::pi << ' ' << trig16::tof(16384)/trig16::pi << std::endl;

   std::cout << trig16::tof(512)/trig16::pi << ' ' << 32*trig16::tof(512)/trig16::pi << ' ' << std::endl;

   std::cout << "\n____________\n"  << std::endl;

   uint16_t mask9 = 511;
   uint16_t mask13 = 15<<9;
   for (int i = 0 /*-trig16::Sin15::NBins*/; i<trig16::Sin15::NBins; ++i) {
     
     int16_t j = i;
     /*
     uint16_t r = j&mask9;
     assert(r<512);
     uint16_t bin = (j&mask13)>>9;
     assert(bin<16);
     auto s = trig16::sinT[bin]*trig16::cos9(r) + trig16::cosT[bin]*trig16::sin9(r);
     auto c = trig16::cosT[bin]*trig16::cos9(r) - trig16::sinT[bin]*trig16::sin9(r);
     float x = bin*trig16::pi64  + trig16::tof(r);
     */

     uint16_t q = (j&trig16::mask);
     int16_t y = j - 16384; // rotate by -pi/2
     int16_t z = j - q;  // move to first quadrant
     assert(z>=0);
     assert(z<16384);
     uint16_t  sw = ((j-8192)>>14)&1;
     z  = (z > 8192) ? 16384 -z : z;

      auto [s,c] = trig16::sincos9(i);
      float x = trig16::tof(i);
     if (std::abs(s-std::sin(x))> 5.e-7f || std::abs(c-std::cos(x))> 5.e-7f) {
       std::cout << j << ' ' << z << ' ' << sw << " : "  << s << ' ' << std::sin(x) << " , " << c << ' ' << std::cos(x) << std::endl;
    }
   }

   return 0;

}
