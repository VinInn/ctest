#include <iostream>
#include <cstddef>
#include <typeinfo>
#include <type_traits>

typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef float __attribute__( ( vector_size( 16 ) , aligned(4) ) ) float32x4a4_t;


float32x4_t load(float * x) {
   return *(float32x4a4_t*)(x);
}


struct A {
  float a;
  float32x4a4_t v;

};

  template<typename V>
  struct VType {
    static auto elem(V x) -> typename std::remove_reference<decltype(x[0])>::type { return x[0];}	      
  };
  template<>
  struct VType<float> {
    static auto elem(float x) -> float { return x;}	      
  };

int main() {

   float v1[3*1025] = {0};
   float v2[3*1025] = {0};
   float32x4_t  z[1024];

   std::cout << typeid(float32x4_t).name() << ' ' <<  sizeof(float32x4_t) << ' ' << alignof(float32x4_t) << std::endl;
   std::cout << typeid(float32x4a4_t).name() << ' ' << sizeof(float32x4a4_t) << ' ' << alignof(float32x4a4_t) << std::endl;
   std::cout << sizeof(A) << ' ' << alignof(A) << ' ' << offsetof(A,a) << ' ' << offsetof(A,v) << std::endl;

    std::cout << typeid(decltype(VType<float32x4_t>::elem(float32x4_t()))).name() << std::endl;
    std::cout << typeid(decltype(VType<float>::elem(float()))).name() << std::endl;
    // std::cout << typeid(std::remove_extent<float32x4_t>::type).name() << std::endl;

   auto xx = z[0]<z[1];
   std::cout << typeid(xx).name() << ' ' << sizeof(xx) << ' ' << alignof(xx) << std::endl;

   A va1[4];
   A va2[5];
   A va3[6];

   for (int i=0; i<4; ++i) {
     std::cout << i << std::endl;
     va3[i+2].v = va1[i].v+va2[i+1].v;
   }


  float32x4_t a{1.,3.,-5.,0};
  for (int i=0; i<(3*1024); i+=3) {
     float32x4a4_t & k1 = *(float32x4a4_t*)((v1+i));
     float32x4a4_t & k2 = *(float32x4a4_t*)((v2+i));
     k1 += a;
     k2 += a + k1; 
  }


  for (int i=0; i<1024; i++) {
     auto k1 = load(v1+3*i);
     auto k2 = load(v2+3*i);
     z[i] = k2+k1;
  }



   std::cout << '\n' << va3[5].v[3] << std::endl;
   std::cout << z[0][0] << ' ' << z[0][3] << std::endl;
   std::cout << z[10][0] << ' ' << z[10][3] << std::endl;
   std::cout << z[1023][0] << ' ' << z[1023][3] << std::endl;
   return 0;
}
