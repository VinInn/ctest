#include <cmath>
#include<cstdio>

#include <x86intrin.h>
#include<tuple>


std::pair<float,float> rsqrt(float x) {
   //float r = 1.0f / __builtin_sqrtf(x);
   float r;
   _mm_store_ss( &r, _mm_rsqrt_ss( _mm_load_ss( &x ) ) );
   float rx = r*x, drx = __builtin_fmaf(r, x, -rx);
   float h = __builtin_fmaf(r,rx,-1.0f) + r*drx, dr = (r*0.5f)*h;
   // fast_two_sum
   auto hi = r - dr;
   auto e = hi - r; /* exact */
   auto lo = -dr - e; /* exact */
   // float rf = r - dr;
   return {hi,lo};
 }


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

   auto qf = rsqrt(k);
   printf("%a %a,%a %a\n",k,qf.first,qf.second, double(qf.first)+double(qf.second));

   double qd = 1./std::sqrt(double(k));
   printf("%a %a\n",k,qd);

   return 0;
}
