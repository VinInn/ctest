#include<cmath>
constexpr unsigned int N=1024;
float a[1024];
float b[1024];
float c[1024];
float d[1024];

void asq() {
#pragma omp simd
  for (auto i=0U;i<1024;++i) {
    b[i] = std::sqrt(a[i]);
  }
}

float sumq() {
  auto s = 0.f;
#pragma omp simd reduction(+:s)
  for (auto i=0U;i<1024;++i) {
    s += std::sqrt(a[i]);
  }
  return s;
}



float sum() {
  auto s = 0.f;
  for (auto i=0U;i<1024;++i) {
    s += a[i]*b[i];
  }
  return s;
}


float sumO() {
  auto s = 0.f;
#pragma omp simd
  for (auto i=0U;i<1024;++i) {
    s += a[i]*b[i];
  }
  return s;
}


float sumO1() {
  auto s = 0.f;
#pragma omp simd reduction(+:s)
  for (auto i=0U;i<1024;++i) {
    s += a[i]*b[i];
  }
  return s;
}


#pragma omp declare simd
float min(float a, float b) { return a < b ? a : b; }
#pragma omp declare simd
float distsq(float x, float y) { return (x - y) * (x - y); }
void example() {
#pragma omp parallel for simd
for (auto i=0U; i<N; i++) { d[i] = min(distsq(a[i], b[i]), c[i]); } }


#pragma omp declare simd
void multSelf(int & a, int b) { a *=b;}

#pragma omp declare simd
unsigned int mult(unsigned int a, unsigned int b) {
  constexpr unsigned int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  unsigned long long  temp = (unsigned long long)(a) * (unsigned long long)(b); // result type is operand's type
  // Rounding; mid values are rounded up
  temp += K;
  // Correct by dividing by base
  return (temp >> Q);  
}



#pragma omp declare simd
void multSelf(unsigned int & a, unsigned int b) { a = mult(a,b);}

#pragma omp declare reduction (foo:int: multSelf(omp_out,omp_in))
#pragma omp declare reduction (foo:unsigned int: multSelf(omp_out,omp_in))


int foo(int const * a, int N) {
  int s=1;
#pragma omp simd aligned(a : 32) reduction(foo:s)
  for (int i=0; i<N; ++i)
    multSelf(s,a[i]);
  return s;
}
	
void bar(unsigned int const * a, unsigned int const *b, unsigned int * c, int N) {
#pragma omp simd aligned(a,b,c : 32)
  for (int i=0; i<N; ++i)
    c[i]=mult(a[i],b[i]);
}
	
unsigned int barRed(unsigned int const * a, int N) {
    unsigned int s=1;
#pragma omp simd aligned(a : 32) reduction(foo:s)
  for (int i=0; i<N; ++i)
    multSelf(s,a[i]);
  return s;
}

