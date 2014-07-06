  inline double ll2d(unsigned long long x) {
    union { double f; unsigned long long i; } tmp;
    tmp.i=x;
    return tmp.f;
  }
  
  
  inline unsigned long long d2ll(double x) {
    union { double f; unsigned long long i; } tmp;
    tmp.f=x;
    return tmp.i;
  }

inline int hl(long long l) {
  union { int v[2]; long long i; } tmp;
    tmp.i=l;
    return tmp.v[0];
}

double a[1024],b[1024];
float c[1024];
long long ll[1024];

unsigned long long l1[1024], l2[1024]; int i3[1024];
void sl() {
  for (int i=0;i!=1024;++i)
    l1[i] = l2[i]<<52;
}

void rl() {
  for (int i=0;i!=1024;++i) {
    l1[i] = (l2[i]>>52) & 0x7ff;
    i3[i]=l1[i];
  }
}

void foo() {
  for (int i=0;i!=1024;++i) {
    unsigned long long n=d2ll(a[i]);
    b[i] = ll2d(n << 52);
  }
}


void bar() {
  for (int i=0;i!=1024;++i) {
    //    unsigned long long n=l1[i];
    unsigned long long n=d2ll(a[i]);
    unsigned long long e = ((n >> 52) & 0x7ff);
    int ie =e;
    c[i] = (ie-1023);
  }
}


void e2() {
  for (int i=0;i!=1024;++i) {
    long long n=d2ll(a[i]);
    n &=0xfffffffffffffL;
    const long long p05 = d2ll(0.5);
    n |= p05;
    b[i] = ll2d(n);
  }
}

/*
#include<iostream>
int main() {
  double d = 1.23e100;
  long long ll = d;
  std::cout << d << std::endl;
  std::cout << ll << std::endl;
  }
*/
