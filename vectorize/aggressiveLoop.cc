#include <cmath>
#include<limits>

float x[1024];
float y[1024];
float w[512];
float z[128];

float c,q;

void foo() {
  for (int i=0; i<1024; ++i) {
   auto zz=z[i];
   auto yy = y[i];
   if(x[i] > q)  yy = zz;
   y[i]=yy;
  }
}

void foo2() {
  for (int i=0; i<1024; ++i) {
   auto zz=z[i];
   auto yy = w[i];
   if(x[i] > q)  yy = zz;            
   x[i]=yy;
  }
}

void foo3() {
  for (int i=0; i<1024; ++i) {
   auto zz=z[i];
   auto yy = w[i];
   if(x[i] > q)  yy = zz;
   w[i]=yy;
  }
}

