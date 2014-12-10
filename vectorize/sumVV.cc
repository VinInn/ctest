#include<cstring>
typedef float __attribute__( ( vector_size( 16 ) ) ) V4;


void sum(float const * a, float const * b, float * c) {
  
  V4 va,vb,vc;
  memcpy(&va,a,16);
  memcpy(&vb,b,16);
  vc = va+vb;
  memcpy(c,&vc,16);
  
}


#include<algorithm>

void vsum(float const * a, float const * b, float * c, int N) {
auto kernel = [&](int i, int k) {
  V4 va,vb,vc;
  memcpy(&va,a+i,4*k);
  memcpy(&vb,b+i,4*k);
  vc = va+vb;
  memcpy(c+i,&vc,4*k);
};
int i=0;
for (; i<N-4 ; i+=4) kernel(i,4);
kernel(i,N-i);

}
