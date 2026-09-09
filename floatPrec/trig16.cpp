

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

   double a[4] = {0.5,2.,-2.,-0.5};
   for ( auto x : a ) {
     auto [s,c] = trig16::sincos(trig16::to16(x));
     std::cout << x << ' ' << trig16::to16(x) << ' ' << std::sin(x)<<',' << std::cos(x)
             << ' ' << s <<',' << c << std::endl;
   }

   std::cout << "\n____________\n"  << std::endl;

    
   std::cout << "sincos14" << std::endl;
   error(trig16::sincos);
   std::cout << std::endl;


   std::cout << "\n____________\n"  << std::endl;

    
   std::cout << trig16::tof(16)/trig16::pi << ' ' << trig16::tof(256)/trig16::pi << ' ' 
             << trig16::tof(512)/trig16::pi << ' ' << trig16::tof(16384)/trig16::pi << std::endl;

   std::cout << trig16::tof(512)/trig16::pi << ' ' << 32*trig16::tof(512)/trig16::pi << ' ' << std::endl;

   uint16_t mask9 = 511;;
   uint16_t mask13 = 15<<9;
   for (int i = -trig16::Sin15::NBins; i<trig16::Sin15::NBins; ++i) {
     int16_t j = i;
     uint16_t r = j&mask9;
     assert(r<1024);
     uint16_t bin = (j&mask13)>>9;
     int u = bin&1; // subtract...
     assert(bin<16);
     int tick = (bin+1)/2;
     auto a = trig16::tick[tick];
     float x = a + ( (u==1) ? -trig16::tof(512-r) : trig16::tof(r));
     assert(x>=0);
     assert(x<=0.25f*trig16::pi);
     if (std::abs(x -trig16::tof(j&8191)) > 1.e-6f) {
        std::cout << j << ' ' << (j&8191) << ' ' << r << ' ' << trig16::tof(r) << ' ' << bin << ' ' << x << ' ' << trig16::tof(j&8191) << std::endl;
     }
     float s,c;
     uint16_t rr = (u==1) ? 512-r : r;
     s = (rr==512) ? trig16::sinPI64 : trig16::sin9(rr);
     c = (rr==512) ? trig16::cosPI64 : trig16::cos9(rr);
     auto s2 = trig16::cosT[tick]*s;
     auto c2 = trig16::sinT[tick]*s;
     s = trig16::sinT[tick]*c +  ( (u==1) ? -s2 : s2);
     c = trig16::cosT[tick]*c +  ( (u==1) ? c2 : -c2);
     if (std::abs(s-std::sin(x))> 5.e-7f || std::abs(c-std::cos(x))> 5.e-7f) {
       std::cout << j << ' ' << (j&8191) << ' ' << r << ' ' << s << ' ' << std::sin(x) << ' ' << c << ' ' << std::cos(x) << std::endl;
    }
   }

   return 0;

}
