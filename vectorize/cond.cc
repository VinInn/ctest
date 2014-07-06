#include <cstdlib>
#include <cstring>
float x[1024];
float y[1024];
float z[1024];
float w[1024];

int k[1024];
int j[1024];


void bar() {
  for (int i=0; i<1024; ++i)
    z[i] = ( (x[i]>0) & (w[i]<0)) ? z[i] : y[i];
}


void barX() {
  for (int i=0; i<1024; ++i) {
    k[i] = x[i]>0;
    k[i] &=  w[i]<y[i];
    z[i] = (k[i]) ? z[i] : y[i];
 }
}

void barMP() {
#pragma omp simd
  for (int i=0; i<1024; ++i)
    z[i] = ( int(x[i]>0) & int(w[i]<0)) ? z[i] : y[i];
}


void barInt() {
  for (int i=0; i<1024; ++i)
    z[i] = ( (int(x[i]>0)<1) & (int(w[i]<0)<1) ) ? z[i] : y[i];
}

void barInt0() {
  for (int i=0; i<1024; ++i)
    z[i] = ( (0+int(x[i]>0)) & (0+int(w[i]<0)) ) ? z[i] : y[i];
}


void barPlus() {
  for (int i=0; i<1024; ++i)
    z[i] = (2==(int(x[i]>0) + int(w[i]<0))) ? z[i] : y[i];
}


void foo() {
  for (int i=0; i<1024; ++i)
    z[i] = ( k[i] & j[i] ) ? z[i] : y[i];
}


void foo2() {
  for (int i=0; i<1024; ++i) {
    k[i] = x[i]>0; j[i] = w[i]<0;
  }
}

void bar2() {
  for (int i=0; i<1024; ++i) {
    k[i] = x[i]>0; j[i] = w[i]<0;
    z[i] = ( k[i] & j[i]) ? z[i] : y[i];
 }
}

