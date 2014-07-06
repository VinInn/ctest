#include<cmath>

const int N=1024;

float __attribute__ ((aligned(16))) a[N];
float __attribute__ ((aligned(16))) b[N];
float __attribute__ ((aligned(16))) c[N];
float __attribute__ ((aligned(16))) d[N];
int __attribute__ ((aligned(16)))   k[N];



float __attribute__ ((aligned(16))) co[12];
float __attribute__ ((aligned(16))) hist[100];


// do not expect GCC to vectorize (yet)
void foo() {
  for (int i=0; i!=N; ++i) {
    float x = co[k[i]];
    float y = a[i]/std::sqrt(x*b[i]);
    ++hist[int(y)];
  } 
}


// let's give it an hand: split the loop so that the "heavy duty one" vectorize
void bar() {
  const int S=8;
  int loops = N/S;
  float x[S];
  float y[S];
  for (int jj=0; jj!=loops; ++jj) {
    int j = jj*S;
    for (int i=0; i!=S; ++i)
      x[i] = co[k[j+i]];
    for (int i=0; i!=S; ++i) // this should vectorize
      y[i] = a[j+i]/std::sqrt(x[i]*b[j+i]);
    for (int i=0; i!=S; ++i)
      ++hist[int(y[i])];
  } 
}



void bar2(int jj) {
  const int S=8;
  float x[S];
  float y[S];
  int j = jj*S;
  for (int i=0; i!=S; ++i)
    x[i] = co[k[j+i]];
  for (int i=0; i!=S; ++i) // this should vectorize
    y[i] = a[j+i]/std::sqrt(x[i]*b[j+i]);
  for (int i=0; i!=S; ++i)
    ++hist[int(y[i])];
} 


void bug() {
  for (int i=1; i!=N; ++i)
    a[i]+=a[i-1];
  for (int i=0; i!=N; ++i)
    b[i]=a[i]+c[i];
}

void bug2() {
  for (int i=1; i!=N; ++i) {
    a[i]+=a[i-1];
    b[i]=a[i]+c[i];
  }
}
