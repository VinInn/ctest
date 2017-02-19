#include <cmath>
#include<limits>

float x[1024];
float y[512];
float z[128];

float c,q;

void foo2() {
  for (int i=0; i<1024; ++i) {
   auto zz=z[i];
   auto yy = y[i];
   auto xx = x[i] > c ? x[i]-c : x[i] - zz;
   if(x[i] > q)  yy =  std::sqrt(xx);
   y[i]=yy;
  }
}


