#include <x86intrin.h>
#ifdef __AVX512F__
#define NATIVE_VECT_LENGH 64
#elif __AVX__
#define NATIVE_VECT_LENGH 32
#else
#define NATIVE_VECT_LENGH 16
#endif

typedef float __attribute__( ( vector_size( NATIVE_VECT_LENGH ) ) ) vfloat32_t;
typedef float __attribute__( ( vector_size( NATIVE_VECT_LENGH ) , aligned(4) ) ) vfloat32a4_t;
typedef int __attribute__( ( vector_size( NATIVE_VECT_LENGH ) ) ) vint32_t;


inline
vfloat32_t load(float const * x) {
   return *(vfloat32a4_t const *)(x);
}


int minloc(float const * x, int N) {
  vfloat32_t v0;
  vint32_t index;

  constexpr int WIDTH = NATIVE_VECT_LENGH/4; 

  auto M = WIDTH*(N/WIDTH);
  for (int i=M; i<N; ++i) {
    v0[i-M] = x[i];
    index[i]=i;
  }
  for (int i=N; i<M+WIDTH;++i) {
    v0[i-M] = x[0];
    index[i]=0;
  }
  vint32_t j;  for (int i=0;i<WIDTH; ++i) j[i]=i;  // can be done better
  for (int i=0; i<M; i+=WIDTH) {
    decltype(auto) v = load(x+i);
    index =  (v<v0) ? j : index;
    v0 = (v<v0) ? v : v0;
    j+=WIDTH;
  }
  auto k = 0;
  for (int i=1;i<WIDTH; ++i) if (v0[i]<v0[k]) k=i;
  return index[k];
}


int lmin(float const *  x, int N) {
  int k=0;
  for (int i=1; i<N; ++i) {
    k =  (x[i] < x[k]) ? i : k;
  }
  return k;
}

#include<iostream>
#include<algorithm>
#include <x86intrin.h>
unsigned int taux=0;
inline unsigned long long rdtscp() {
 return __rdtscp(&taux);
}


int main() {

   std::cout << "NATIVE_VECT_LENGH " << NATIVE_VECT_LENGH << std::endl;
  
  int N = 1024*4;
  float x[N];
  for (int i=0; i<N; ++i) x[i]= i%2 ? i : -i;
  for (int i = 0; i<10; ++i) {
   std::random_shuffle(x,x+N);
   long long ts = -rdtscp();
   int l1 = std::min_element(x+i,x+N) - (x+i);
   ts +=rdtscp();
   long long tv = -rdtscp();	
   int l2 = minloc(x+i,N-i);
   tv +=rdtscp();

    std::cout << "min is at " << l1 << ' ' << ts << std::endl;
    std::cout << "minloc " << l2 << ' ' << tv << std::endl;
  }
  return 0;

}
