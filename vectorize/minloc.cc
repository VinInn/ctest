#include <x86intrin.h>
#include<tuple>

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

  template<typename V>
  bool testz(V const t) {
   return _mm256_testz_si256(__m256i(t),__m256i(t));
  }
  
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
    auto mask = v<v0;
#ifdef DOTESTZ
    if (!testz(mask))
#endif 
    {
      index =  mask ? j : index;
      v0 = mask ? v : v0;
    }
    j+=WIDTH;
  }
  auto k = 0;
  for (int i=1;i<WIDTH; ++i) if (v0[i]<v0[k]) k=i;
  return index[k];
}


template<typename T>
int lmin(T const *  x, int N) {
  int k=0;
  auto m = x[k];
  for (int i=1; i<N; ++i) {
    auto d = x[i];
    if (d<m) {
      k = i; m=d;
    }
  }
  return k;
}


// find location of close "point" 
template<typename T, typename C>
std::tuple<int,T> closestloc(T const * x, T const * y, int N, C c) {
  using namespace minlocDetails;
  using vfloat_t = typename Traits<T>::vfloat_t;
  using vint_t = typename Traits<T>::vint_t;
  constexpr int TSIZE = sizeof(T);
  vfloat_t v0,x0,y0;
  vint_t index;

  constexpr int WIDTH = NATIVE_VECT_LENGH/TSIZE; 

  auto M = WIDTH*(N/WIDTH);
  for (int i=M; i<N; ++i) {
    x0[i-M] = x[i];
    y0[i-M] = y[i];
    index[i]=i;
  }
  for (int i=N; i<M+WIDTH;++i) {
    x0[i-M] = x[0];
    y0[i-M] = y[0];
    index[i]=0;
  }

  v0 = c(x0,y0);
  
  vint_t j;  for (int i=0;i<WIDTH; ++i) j[i]=i;  // can be done better
  for (int i=0; i<M; i+=WIDTH) {
    decltype(auto) vx = load(x+i);
    decltype(auto) vy = load(y+i);
    auto v = c(vx,vy);
    index =  (v<v0) ? j : index;
    v0 = (v<v0) ? v : v0;
    j+=WIDTH;
  }
  auto k = 0;
  for (int i=1;i<WIDTH; ++i) if (v0[i]<v0[k]) k=i;
  
  return std::make_tuple(index[k],v0[k]);
}


template<typename T, typename C>
std::tuple<int,T> closest(T const * x, T const * y, int N, C c) {
  int k=0;
  T  m =c(x[0],y[0]);
  for (int i=1; i<N; ++i) {
    auto d = c(x[i],y[i]);
    if (d<m) {
      k = i; m=d;
    }
  }
  return std::make_tuple(k,m);
}



#include<iostream>
#include<algorithm>
#include <x86intrin.h>
unsigned int taux=0;
inline unsigned long long rdtscp() {
 return __rdtscp(&taux);
}

#include<cassert>
template<typename T>
int go() {
  using namespace minlocDetails;

  std::cout << "NATIVE_VECT_LENGH " << NATIVE_VECT_LENGH << std::endl;

  long long tt[4]={0};
  int N = 1024*4+3;
  T x[N],y[N];
  for (int kk=0;kk<3;++kk) {
    for (int i=0; i<N; ++i) x[i]= i%2 ? i : -i;
    for (int i=0; i<N; ++i) y[i]= i%2 ? i : -i;
    for (int i = kk; i<10+kk; ++i) {
      std::random_shuffle(x,x+N);
      std::random_shuffle(y,y+N);
      if (kk==0 && i==0) x[i]=-N*N;
      if (kk==1 && i==0) x[N]=-N*N;
      long long ts = -rdtscp();
      int l1 = lmin(x+i,N-i); // std::min_element(x+i,x+N) - (x+i);
      ts +=rdtscp();
      long long tv = -rdtscp();	
      int l2 = minloc(x+i,N-i);
      tv +=rdtscp();

      T x0=0, y0=0;
      if (kk==0 && i==0){x[i]=x0; y[i]=y0;}
      if (kk==1 && i==0){x[N]=x0; y[N]=y0;}

      auto dist = [&](auto x, auto y) { return (x-x0)*(x-x0)+(y-x0)*(y-x0);};
      long long tsn = -rdtscp();
      auto l3 = closest(x+i,y+i,N-i,dist);
      tsn +=rdtscp();
      long long tvn = -rdtscp();	
      auto l4 = closestloc(x+i,y+i,N-i,dist);
      tvn +=rdtscp();

      
      if (kk!=0) { tt[0]+=ts; tt[1]+=tv;  tt[2]+=tsn; tt[3]+=tvn;}
      if(kk==2) {
	std::cout << "min is at " << l1 << ' ' << ts << std::endl;
	std::cout << "minloc " << l2 << ' ' << tv << std::endl;
	std::cout << "closest is at " << std::get<0>(l3) << ' ' << tsn << std::endl;
	std::cout << "closest " << std::get<0>(l4) << ' ' << tvn << std::endl;
      }
      if(l1!=l2) std::cout << x[l1] << ' ' << x[l2] << std::endl;
      if(l3!=l4) std::cout << std::get<1>(l3) << ' ' << std::get<1>(l4) << std::endl;
    }
  }
  std::cout << "times " << tt[0] << ' ' << tt[1] << ' ' << tt[2] << ' ' << tt[3] << ' ' << std::endl;
  return N;

}

int main() {
  return go<float>() + go<double>();
   
}
