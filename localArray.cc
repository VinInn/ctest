#include <ext/array_allocator.h>
#include <map>
#include<algorithm> 

void foo(float *);
void cpy(float *, float const *, int);

float bar(int n, float const * a) {
   float loc1[100];
   for ( float & x : loc1) x+=1.3;
   float loc2[n];  
   for ( float & x : loc2) x+=1.3;
   std::for_each(loc2,loc2+n,[](decltype(*loc2)&&x){x+=1.3;});
   cpy(loc1,a,n);
   foo(loc2);
   return loc1[3]+loc2[2];
}
