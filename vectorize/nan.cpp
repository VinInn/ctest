#include<type_traits>
#include <cmath>

 template<typename V1,typename V2 >
inline
auto dot2(V1  x, V2 y) ->typename  std::remove_reference<decltype(x[0])>::type {
   typedef typename std::remove_reference<decltype(x[0])>::type T;
  T ret=0;
  for (int i=0;i!=2;++i) ret+=x[i]*y[i];
  return ret;
}

typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;



void set (float32x4_t & v, float x) {
  float32x4_t vv{x,x,x,x};
  v = vv;
}



float get() {
  return std::sqrt(-1);
}



float val2(float32x4_t v) {
  return ::dot2(v,v);
}


float val(float32x4_t v) {
  return std::sqrt(val2(v));
}





#include <iostream> 
int main(int q, char**) {

 float32x4_t v;
 if (q>2) 
   set(v,1.f);
 else
   set(v,get());

 double pt = val(v);
 std::cout << pt << ' ' << val2(v) << ' ' << val(v) << std::endl;
 if (pt == 0) std::cout << "zero pt " << pt << std::endl;

};
