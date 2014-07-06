typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef double __attribute__( ( vector_size( 32 ) ) ) float64x4_t;

typedef int __attribute__( ( vector_size( 16 ) ) ) int32x4_t;
typedef unsigned int __attribute__( ( vector_size( 16 ) ) ) uint32x4_t;




uint32x4_t cast(int32x4_t i) {
  return  uint32x4_t(i);
}

float32x4_t convert(int32x4_t i) {
  return  float32x4_t(i);
}

int32x4_t convert(float32x4_t f) {
  return  int32x4_t(f);
}



#include<iostream>
template<typename V>
void print(V v) { std::cout << v[0] <<','<< v[1] <<',' << v[2] <<',' << v[3] << std::endl;}

int main() {
   float32x4_t vf{4.2,-3.2, 1.2, 7.2};

   print(vf);
   
   print(convert(vf));

   std::cout << int(vf[0]) << ' ' << *reinterpret_cast<int*>(&vf[0]) << std::endl;

   int32x4_t vi{int(vf[0]),int(vf[1]),int(vf[2]),int(vf[3])};
   print(vi);   

  return 0;
}


