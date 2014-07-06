template<typename T, int N> 
struct ExtVecTraits {
typedef T __attribute__( ( vector_size( N*sizeof(T) ) ) ) type;
  static type add(type x, type y) { return x+y;}
};


template<typename T, int N> using ExtVec =  typename ExtVecTraits<T,N>::type; 
//template<typename T, int N> using ExtVec = T __attribute__( ( vector_size( N*sizeof(T) ) ) );

template<typename T> using Vec4 = ExtVec<T,4>; 


using V4 = Vec4<float>;


template<typename T>
inline
T add(T x, T y) {
  return x+y;
} 

/*
template<typename T, int N>
inline
ExtVec<T,N> add(ExtVec<T,N> x, ExtVec<T,N>  y) {
  return x+y;
}
*/



V4 addf(V4 x, V4 y) {
return add(x,y);
}


V4 sum(V4 x, V4 y) {
return x+y;
}

#include<iostream>
template<typename T>
void p( T v) {
  std::cout <<sizeof(v) << std::endl;
  std::cout <<sizeof(v[0]) << std::endl;
  auto t=v[0]; t=0;
  std::cout <<sizeof(t) << std::endl;
}


int main() {
  p((V4){1});
  p(ExtVec<long long,8>{-1LL});

}
