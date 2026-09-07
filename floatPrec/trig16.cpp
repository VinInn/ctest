#include "trig16.h"


#include<iostream>
#include<cmath>
#include<algorithm>

int main() {

   std::cout << trig16::pi4d << ' ' << trig16::mask << ' ' << trig16::cto16 << std::endl;

   LUT<14> sin14(std::sin<double>,trig16::pi2d);
   std::cout << sin14(0) << ' ' << sin14(LUT<14>::NBins-1) << std::endl;
   std::cout << sin14(1) << ' ' << sin14(LUT<14>::NBins-2) << std::endl;

   double a[4] = {0.5,2.,-2.,-0.5};
   for ( auto x : a ) {
     auto [s,c] = trig16::sincos(trig16::to16(x));
     std::cout << x << ' ' << trig16::to16(x) << ' ' << std::sin(x)<<',' << std::cos(x)
             << ' ' << s <<',' << c << std::endl;
   }

   float emaxS = 0; int lS=0;
   float emaxC = 0; int lC=0;
   for (int i = -LUT<15>::NBins; i<LUT<15>::NBins; ++i) {
     float x = trig16::tof(i);
     auto [s,c] = trig16::sincos(i);
     auto dS = std::abs(s-std::sin(x));
     auto dC = std::abs(c-std::cos(x));
     if (dS>emaxS) { emaxS=dS; lS=i;}
     if (dC>emaxC) { emaxC=dC; lC=i;}
   }
   std::cout << "emax " << trig16::tof(lS) <<':'<< lS <<':'<< emaxS 
            << " , "  <<   trig16::tof(lC) <<':' << lC <<':'<< emaxC << std::endl;
   return 0;

}
