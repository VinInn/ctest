typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef float __attribute__( ( vector_size( 16 ) , aligned(4) ) ) float32x4a4_t;
typedef int __attribute__( ( vector_size( 16 ) ) ) int32x4_t;


inline
float32x4_t load(float const * x) {
   return *(float32x4a4_t const *)(x);
}


int minloc(float const * x, int N) {
  float32x4_t v0;
  int32x4_t index;

  auto M = 4*(N/4);
  for (int i=M; i<N; ++i) {
    v0[i-M] = x[i];
    index[i]=i;
  }
  for (int i=N; i<M+4;++i) {
    v0[i-M] = x[0];
    index[i]=0;
  }
  int32x4_t j = {0,1,2,3};
  for (int i=0; i<M; i+=4) {
    decltype(auto) v = load(x+i);
    index =  (v<v0) ? j : index;
    v0 = (v<v0) ? v : v0;
    j+=4;
  }
  auto k = 0;
  for (int i=1;i<4; ++i) if (v0[i]<v0[k]) k=i;
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

  float x[1024];
  for (int i=0; i<1024; ++i) x[i]= i%2 ? i : -i;
  for (int i = 0; i<10; ++i) {
   std::random_shuffle(x,x+1024);
   long long ts = -rdtscp();
   int l1 = std::min_element(x+i,x+1024) - (x+i);
   ts +=rdtscp();
   long long tv = -rdtscp();	
   int l2 = minloc(x+i,1024-i);
   tv +=rdtscp();

    std::cout << "min is at " << l1 << ' ' << ts << std::endl;
    std::cout << "minloc " << l2 << ' ' << tv << std::endl;
  }
  return 0;

}
