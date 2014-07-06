typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef int __attribute__( ( vector_size( 16 ) ) ) int32x4_t;

#include<cassert>
int main() {

  float a = 3.14f;
  // float32x4_t va = float32x4_t{} + a;
  float32x4_t va = a;
  int32x4_t t = va==a;

  for (int i=0; i!=4;++i) {
    assert(va[i]==a);
    assert(t[i]==-1);
  }

  return 0;
}
