template<typename P, typename T>
void assignSoA(P * p, T * & t) {
  static_assert(sizeof(P)==sizeof(T));
  t = reinterpret_cast<T*>(p);
}

template<typename P>
void assignSoA(int stride, P * p) {
}

template<typename P, typename T, typename ... Tail>
void assignSoA(int stride, P * p, T * & t, Tail& ... tail) {
  assignSoA(p,t);
  assignSoA(stride, p+stride,tail...);
}


#include<iostream>

int main() {

   int v[100];
   
   float * x;
   float * y;
   int * z;

   assignSoA(v,z);
   
   std::cout << v << ' ' << z <<std::endl;

  assignSoA(10, v,x,y,z);

   std::cout << v << ' ' << x << ' ' << y-x << ' ' << z-v <<std::endl;


   return 0;
}
