#include <cmath>
#include<cstdio>

#include <x86intrin.h>

int main(int n, char * v[]) {
   float k = float(n)*0.0118255867f;

   float y;
   _mm_store_ss( &y, _mm_rsqrt_ss( _mm_load_ss( &k ) ) );
   printf("%a %a\n",k,y);

   float q = 1.f/std::sqrt(k);
   printf("%a %a\n",k,q);
   return 0;
}
