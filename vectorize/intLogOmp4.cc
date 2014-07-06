#include <cstdint>
#include<cmath>
union binary32 {
  binary32() : ui32(0) {};
  binary32(float ff) : f(ff) {};
  binary32(int32_t ii) : i32(ii){}
  binary32(uint32_t ui) : ui32(ui){}
  
  uint32_t ui32; /* unsigned int */                
  int32_t i32; /* Signed int */                
  float f;
};

inline
void irex(float x, int & er, unsigned int & mr) {

  binary32 xx;
  xx.f = x;
  
  er = (((xx.ui32) >> 23) & 0xFF) -127; // extract exponent
  mr = (xx.ui32 & 0x007FFFFF) | 0x00800000; // extract mantissa as an integer number

}

inline
unsigned int mult(unsigned int a, unsigned int b) {
  constexpr unsigned int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  unsigned long long  temp = (unsigned long long)(a) * (unsigned long long)(b); // result type is operand's type
  // Rounding; mid values are rounded up
  temp += K;
  // Correct by dividing by base
  return (temp >> Q);  
}

struct IExMa {
  IExMa(float r=1.f) { irex(r,er,mr); }


  //unsigned long long me;
  int er=0; 
  unsigned int mr=0;

#pragma omp declare simd
  IExMa & operator*=(IExMa b) {
    auto k = mult(mr,b.mr);
    unsigned int ov = k >> 31;  mr = k>>ov;
    er += b.er+ov;
    return *this;
  }

#pragma omp declare simd
  IExMa & reduce(IExMa b) {
    mr = mult(mr>>4,b.mr>>4);
    er += b.er+8;
    return *this;
  }
  
};


#pragma omp declare reduction (foo:struct IExMa: omp_out.reduce(omp_in))

float intLog(float * r, int NN) { 
  // we use this to evaluate the sum of logs taking care, as usual, not to overflow the product of mantissas
  IExMa isp;
  // #pragma omp simd aligned(r : 32) reduction(foo:isp)
  // #pragma omp simd reduction(foo:isp)
#pragma omp parallel for 
  for (int i=0;i<NN;++i) {
     IExMa l(r[i]);
     isp*=l;
  }
  return isp.er+std::log2(isp.mr)-23;
}



#include<iostream>
int main() {
  constexpr int NN=1024;
  alignas(32) float r[1024];

  for ( auto &s : r) s=0.5f;

  std::cout << intLog(r,NN) << std::endl;

  return 0;
}

