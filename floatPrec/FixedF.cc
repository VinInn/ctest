inline
unsigned int mult(unsigned int a, unsigned int b) {
  typedef unsigned long long ull; // (to support >>)
  // a and b are of the form 1.m with m of Q bits  as int is therefore max 2^(Q+2)-1. a*b is therefore < 2^(2*(Q+2)) 
  constexpr int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  ull  temp = (ull)(a) * (ull)(b); // result type is operand's type
  // Rounding; mid values are rounded up
  temp += K;
  // Correct by dividing by base   
  return (temp >> Q);  
}

inline
unsigned long long multL(unsigned long long a, unsigned long long b) {
  typedef unsigned long long ull; // (to support >>)
  // a and b are of the form 1.m with m of Q bits. As int is therefore max 2^(Q+2)-1. a*b is therefore < 2^(2*(Q+2)) 
  constexpr int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  ull  temp = (ull)(a) * (ull)(b); 
  // Rounding; mid values are rounded up
  temp += K;
  // Correct by dividing by base   
  return temp >> Q;  
  // return (temp/(2*K));  
}

#include <x86intrin.h>

typedef unsigned long long __attribute__( ( vector_size( 32 ) ) ) uint64x4_t;
typedef signed long long __attribute__( ( vector_size( 32 ) ) ) int64x4_t;
typedef signed int __attribute__( ( vector_size( 32 ) ) ) int32x8_t;
typedef unsigned int __attribute__( ( vector_size( 32 ) ) ) uint32x8_t;
typedef float __attribute__( ( vector_size( 32 ) ) ) float32x8_t;

typedef unsigned long long __attribute__( ( vector_size( 16 ) ) ) uint64x2_t;
typedef signed long long __attribute__( ( vector_size( 16 ) ) ) int64x2_t;
typedef signed int __attribute__( ( vector_size( 16 ) ) ) int32x4_t;
typedef unsigned int __attribute__( ( vector_size( 16 ) ) ) uint32x4_t;
typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;


union binary128 {
  binary128() : ul{0,0} {};
  binary128(float32x4_t ff) : f(ff) {};
  binary128(int32x4_t ii) : i(ii){}
  binary128(uint32x4_t ii) : ui(ii){}
  binary128(int64x2_t ii) : l(ii){}
  binary128(uint64x2_t ii) : ul(ii){}
  
  __m128i i128;
  float32x4_t f;
  int32x4_t i;
  uint32x4_t ui;
  int64x2_t l;
  uint64x2_t ul;
};

union binary256 {
  binary256() : ul{0,0} {};
  binary256(float32x8_t ff) : f(ff) {};
  binary256(int32x8_t ii) : i(ii){}
  binary256(uint32x8_t ii) : ui(ii){}
  binary256(int64x4_t ii) : l(ii){}
  binary256(uint64x4_t ii) : ul(ii){}
  
  __m256i i256;
  float32x8_t f;
  int32x8_t i;
  uint32x8_t ui;
  int64x4_t l;
  uint64x4_t ul;
};


__uint128_t mult(__uint128_t a, __uint128_t b) {
  return a*b;
}

__uint128_t multD(__uint128_t a, __uint128_t b) {
  constexpr int Q = 52;
  constexpr unsigned long long K  = (1UL << (Q-1));
  auto temp = a*b;
  temp += K;
  return temp >> Q;
}




uint64x4_t mult(uint64x4_t a,uint64x4_t b) {
  constexpr int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  uint64x4_t temp = a*b;
  temp += K;
  return temp >> Q;
}


uint64x4_t multI(uint64x4_t a,uint64x4_t b) {
  constexpr int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  uint64x4_t temp =  uint64x4_t(_mm256_mul_epu32(__m256i(a),__m256i(b)));
  temp += K;
  return temp >> Q;
}

uint64x2_t multI(uint64x2_t a,uint64x2_t b) {
  constexpr int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  uint64x2_t temp =  uint64x2_t(_mm_mul_epu32(__m128i(a),__m128i(b)));
  temp += K;
  return temp >> Q;
}


void irex(float32x4_t x, int32x4_t & er, uint32x4_t & mr) {
 
  binary128 xx;
  xx.f = x;
  
  // as many integer computations as possible, most are 1-cycle only, and lots of ILP.
  er = (((xx.i) >> 23) & 0xFF) -127; // extract exponent
  mr = (xx.ui & 0x007FFFFF) | 0x00800000; // extract mantissa as an integer number

}

void irex(float32x8_t x, int32x8_t & er, uint32x8_t & mr) {
 
  binary256 xx;
  xx.f = x;
  
  // as many integer computations as possible, most are 1-cycle only, and lots of ILP.
  er = (((xx.i) >> 23) & 0xFF) -127; // extract exponent
  mr = (xx.ui & 0x007FFFFF) | 0x00800000; // extract mantissa as an integer number

}


uint32x4_t multI(uint32x4_t a, uint32x4_t b) {

  binary128 temp1, temp2;
  constexpr int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  temp1.i128 = _mm_mul_epu32(__m128i(a),__m128i(b));
  temp1.ul += K;  temp1.ul >>= Q;  // results are in position 0 and 2

  constexpr int32x4_t mask{1,0,3,2};
  a = __builtin_shuffle(a,mask);
  b = __builtin_shuffle(b,mask);
  temp2.i128 = _mm_mul_epu32(__m128i(a),__m128i(b));
  temp2.ul += K;  temp2.ul >>= Q;

  temp2.ul <<= 32;  // results are now in position 1 and 3
  temp1.ul |=temp2.ul;  

  /*
  constexpr int32x4_t mask2{0,4,2,6};
  temp.i = __builtin_shuffle(temp1.i,temp2.i,mask2);
  */
  return temp1.ui;
}

uint32x8_t multI(uint32x8_t a, uint32x8_t b) {

  binary256 temp1, temp2;
  constexpr int Q = 23;
  constexpr unsigned long long K  = (1 << (Q-1));
  temp1.i256 = _mm256_mul_epu32(__m256i(a),__m256i(b));
  temp1.ul += K;  temp1.ul >>= Q; // results are in position 0,2...

  constexpr int32x8_t mask{1,0,3,2,5,4,8,7};
  a = __builtin_shuffle(a,mask);
  b = __builtin_shuffle(b,mask);
  temp2.i256 = _mm256_mul_epu32(__m256i(a),__m256i(b));
  temp2.ul += K;  temp2.ul >>= Q;

  temp2.ul <<= 32;  // results are now in position 1,3...
  temp1.ul |=temp2.ul;  

  /*
  constexpr int32x8_t mask2{0,8,2,10,4,12,6,14};
  temp1.i = __builtin_shuffle(temp1.i,temp2.i,mask2);
  */
  return temp1.ui;
}



unsigned int   a[1024];
unsigned int   b[1024];
unsigned int   c[1024];

unsigned long long   al[1024];
unsigned long long   bl[1024];
unsigned long long   cl[1024];


void foo() {
 for (int i=0;i!=1024;++i)
   c[i] = mult(a[i],b[i]);
}


unsigned int red() {
  unsigned int s=1;
  for (int i=0;i!=1024;++i)
    s = mult(s,b[i]);
  return s;
}


unsigned int redV() {
  uint32x4_t s{1,1,1,1};
  for (int i=0;i!=1024;i+=4) {
    uint32x4_t bv{b[i+0],b[i+1],b[i+2],b[i+3]};
    s = multI(s,bv);
  }
  return mult(mult(s[0]>>4,s[1]>>4)>>4,mult(s[2]>>4,s[3]>>4)>>4);
}


unsigned int redExp() {
  unsigned int s[4]{1,1,1,1};
  for (int i=0;i!=1024;i+=4)
    for (int j=0; j!=4; ++j)
      s[j] = mult(s[j],b[i+j]);
  return mult(mult(s[0],s[1]),mult(s[2],s[3]));
}

unsigned long long redExpl() {
  unsigned long long s[4]{1,1,1,1};
  for (int i=0;i!=1024;i+=4)
    for (int j=0; j!=4; ++j)
      s[j] = multL(s[j],bl[i+j]);
  return multL(multL(s[0],s[1]),multL(s[2],s[3]));
}


unsigned long long redL() {
  unsigned long long s=1;
  for (int i=0;i!=1024;++i)
    s = multL(s,b[i]);
  return s;
}


unsigned int prod() {
  unsigned int s=1;
  for (int i=0;i!=1024;++i) {
    s = s*b[i]; s+=1024;
  }
  return s;
}

unsigned int prod0() {
  unsigned int s=1;
  for (int i=0;i!=1024;++i) {
    s = s*b[i] + 1024;
  }
  return s;
}



unsigned long long prodL() {
  unsigned long long s=1;
  for (int i=0;i!=1024;++i) {
    s = s*bl[i]; s+=1024UL;
  }
  return s;
}


unsigned long long prodL2() {
  unsigned long long s=1;
  for (int i=0;i!=1024;++i) {
    unsigned long long t = s*bl[i]; t+=1024UL; s=t;
  }
  return s;
}
