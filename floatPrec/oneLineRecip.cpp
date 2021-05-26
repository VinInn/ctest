#include <cmath>
#include<cstdio>

#include <x86intrin.h>

int main(int n, char * v[]) {
   float k = float(1-n) + 0x1.99999ep-4; // 0.1f;
   // float k = float(n)*0.0118255867f;

   float y;
   _mm_store_ss( &y, _mm_rsqrt_ss( _mm_load_ss( &k ) ) );
   printf("%a %a\n",k,y);
   y =  y * (1.5f - 0.5f * k * (y * y));
   printf("%a %a\n",k,y);

   float q = 1.f/std::sqrt(k);
   printf("%a %a\n",k,q);
   return 0;
}
