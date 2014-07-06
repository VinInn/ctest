#include<algorithm>
#define Type int

struct TwoInt {
  unsigned long long me;
  Type a=0;
  Type b=0;

#pragma omp declare simd
  TwoInt & operator+=(TwoInt rh) {
    a+=rh.a;
    b-=rh.b;
  }

#pragma omp declare simd
  TwoInt & add(TwoInt rh) {
    a+=rh.a;
    b-=rh.b;
    return *this;
  }


#pragma omp declare simd
  TwoInt & reduce(TwoInt rh) {
    a+=rh.a;
    b+=rh.b;
    return *this;
  }


};

#pragma omp declare reduction (foo:struct TwoInt: omp_out.reduce(omp_in))


TwoInt sum(Type const * q, int NN) {
  TwoInt s;
#pragma omp parallel for  reduction(foo:s)
  // #pragma omp parallel for simd  aligned(q: 16) reduction(foo:s)
  for (int i=0;i<NN;++i) {
    TwoInt l; l.a=q[i]; l.b = q[i];
    s.add(l);
  }
  return s;
}

TwoInt sum4(Type const * q, int NN) {
  TwoInt s[4];
  for (int i=0;i<NN;i+=4) {
    for (int j=0;j<std::min(4,NN-i);++j) {
      TwoInt l; l.a=q[i+j]; l.b = q[i+j];
      s[j].add(l);
    }
  }
  s[0].reduce(s[1]); s[3].reduce(s[2]); s[3].reduce(s[0]);
  return s[3];
}


#include<iostream>
int main() {
  constexpr int NN=1024;
  Type q[NN];
  Type a=0;
  for (auto & e: q) e=a++;

  auto s = sum(q,NN);
  std::cout << s.a << "," << s.b << std::endl;
  s = sum4(q,NN);
  std::cout << s.a << "," << s.b << std::endl;

  return 0;
}


