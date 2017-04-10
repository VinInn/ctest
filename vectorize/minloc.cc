#include <x86intrin.h>


namespace minlocDetails {

#ifdef __AVX512F__
  constexpr unsigned int NATIVE_VECT_LENGH=64;
#elif __AVX__
  constexpr unsigned int NATIVE_VECT_LENGH=32;
#else
  constexpr unsigned int NATIVE_VECT_LENGH=16;
#endif

  template<typename T> struct Traits{};

  template<> struct Traits<float>{
    typedef float __attribute__( ( vector_size( NATIVE_VECT_LENGH ) ) ) vfloat_t;
    typedef float __attribute__( ( vector_size( NATIVE_VECT_LENGH ) , aligned(4) ) ) vfloatAN_t;
    typedef int __attribute__( ( vector_size( NATIVE_VECT_LENGH ) ) ) vint_t;
  };

  template<> struct Traits<double>{
    typedef double __attribute__( ( vector_size( NATIVE_VECT_LENGH ) ) ) vfloat_t;
    typedef double __attribute__( ( vector_size( NATIVE_VECT_LENGH ) , aligned(8) ) ) vfloatAN_t;
    typedef long long __attribute__( ( vector_size( NATIVE_VECT_LENGH ) ) ) vint_t;
  };
  
  template<typename T>
  inline
  typename Traits<T>::vfloat_t load(T const * x) {
    return *(typename Traits<T>::vfloatAN_t const *)(x);
  }
}

template<typename T>
int minloc(T const * x, int N) {
  using namespace minlocDetails;
  using vfloat_t = typename Traits<T>::vfloat_t;
  using vint_t = typename Traits<T>::vint_t;
  constexpr int TSIZE = sizeof(T);
  vfloat_t v0;
  vint_t index;

  constexpr int WIDTH = NATIVE_VECT_LENGH/TSIZE; 

  auto M = WIDTH*(N/WIDTH);
  for (int i=M; i<N; ++i) {
    v0[i-M] = x[i];
    index[i]=i;
  }
  for (int i=N; i<M+WIDTH;++i) {
    v0[i-M] = x[0];
    index[i]=0;
  }
  vint_t j;  for (int i=0;i<WIDTH; ++i) j[i]=i;  // can be done better
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


template<typename T>
int lmin(T const *  x, int N) {
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


template<typename T>
int go() {
  using namespace minlocDetails;

  std::cout << "NATIVE_VECT_LENGH " << NATIVE_VECT_LENGH << std::endl;
  
  int N = 1024*4+3;
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
  return N;

}

int main() {
  return go<float>() + go<double>();
   
}
