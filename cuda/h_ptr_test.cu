#include "h_ptr.h"
#include<iostream>
#include<cassert>

struct T{int a;};

struct V {
  h_ptr<T> tref;
};

__global__ void set(T * t, V* v) {
 t->a=5;
 v->tref = t;
 
}


int main() {

  relocationTable = new std::unordered_map<unsigned long long,unsigned long long>;


 T * gt;
 V * gv;
 cudaMalloc(&gt,sizeof(T));
 cudaMalloc(&gv,sizeof(V));
 cudaMemset(gv,0,sizeof(V));
 set<<<1,1>>>(gt,gv);
 T ht;
 V hv;
 cudaMemcpy(&hv,gv,sizeof(V),cudaMemcpyDeviceToHost);
 assert(nullptr == hv.tref.get());
 cudaMemcpy(&ht,gt,sizeof(T),cudaMemcpyDeviceToHost);
 assert(nullptr == hv.tref.get());
 (*relocationTable)[(unsigned long long)gt] = (unsigned long long)(&ht);
 assert((&ht) == hv.tref.get());
 assert(5 == hv.tref.get()->a);

 cudaDeviceSynchronize(); 

 // this fake cpu wf
 T ct;
 V cv;
 memset(&cv,0,sizeof(V)); 
 ct.a=5;
 cv.tref = &ct; 
 // better to be ok
 assert((&ct) == cv.tref.get());
 assert(5 == cv.tref.get()->a);


 return 0;

}
