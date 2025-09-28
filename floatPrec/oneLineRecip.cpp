#include <cmath>
#include<cstdio>
#include<cstring>

#include <x86intrin.h>
#include<tuple>



std::pair<float,float> rsqrt(float x) {
   //float r = 1.0f / __builtin_sqrtf(x);
   float r;
   _mm_store_ss( &r, _mm_rsqrt_ss( _mm_load_ss( &x ) ) );
   // standard one NR iteration
   r =  r * (1.5f - 0.5f * x * (r * r));
   float rx = r*x, drx = __builtin_fmaf(r, x, -rx);
   float h = __builtin_fmaf(r,rx,-1.0f) + r*drx, dr = (r*0.5f)*h;
   // fast_two_sum
   auto hi = r - dr;
   auto e = hi - r; /* exact */
   auto lo = -dr - e; /* exact */
   // float rf = r - dr;
   return {hi,lo};
 }



std::pair<float,double>
ulp_error (float x)
{
 auto y = rsqrt (x);
 double z = 1./sqrt(double(x));
 float zf = z;
 float errf = fabsf (y.first - zf);
 double zdw = double(zf) + float(z-double(zf));
 double errd = fabs((double(y.first)+double(y.second))-zdw);
 int ef,ed;
 auto q = frexpf (zf, &ef);
 auto w = frexp (zdw, &ed);
 /* ulp(z) = 2^(e-24) */
 return {ldexpf (errf, 24 - ef),ldexp(errd, 48 - ed)};
}

std::pair<int,int>
iulp_error (float x)
{
 auto y = rsqrt (x);
 double z = 1./sqrt(double(x));
 float hi = z;
 float lo = float(z-double(hi));
 int ih, il, dh,dl;
 memcpy(&ih,&y.first,sizeof(float));
 memcpy(&il,&y.second,sizeof(float));
 memcpy(&dh,&hi,sizeof(float));
 memcpy(&dl,&lo,sizeof(float));
 return {ih-dh,il-dl};
}


void merr() {
  float x, maxerr = 0;
  double med=0;
   x = 0.0118255867f;
   // x = 0x1.fffffcp+1;
   auto r = ulp_error (x);
   printf ("x=%a errf=%f\n", x, r.first);
   printf ("x=%a errd=%f\n", x, r.second);
 for (x = 0.0001f; x < 10000.0f; x = nextafterf (x, 100000.f))
 {
   auto r = ulp_error (x);
   if (r.first > maxerr)
     printf ("x=%a errf=%f\n", x, maxerr = r.first);
   if (r.second > med)
     printf ("x=%a errd=%f\n", x, med = r.second);
  }
}


int main(int n, char * w[]) {
   float a1 = float(1-n) + 0x1.99999ep-4; // 0.1f;
   float a2 = float(n)*0.0118255867f;

   float v[] = {0,0,0x1.04458p+0,0x1.13e07p+1,0x1.fffffcp+1,0x1.f02102p-13};
   v[0] = a1;v[1]=a2;

for ( auto k : v) {

   float y;
   _mm_store_ss( &y, _mm_rsqrt_ss( _mm_load_ss( &k ) ) );
   printf("intr %a %a\n",k,y);
   y =  y * (1.5f - 0.5f * k * (y * y));
   printf("1NR %a %a\n",k,y);

   float q = 1.f/std::sqrt(k);
   printf("libm %a %a\n",k,q);

   auto qf = rsqrt(k);
   printf("me %a {%a,%a} = %a\n",k,qf.first,qf.second, double(qf.first)+double(qf.second));

   double qd = 1./std::sqrt(double(k));
   printf("double %a %a  %a %a\n",k,qd, float(qd)-qf.first, qd-(double(qf.first)+double(qf.second)));

   auto r = ulp_error (k);
   printf ("x=%a errf=%f  ", k, r.first);
   printf (" errd=%f\n", r.second);
   auto ir = iulp_error(k);
   printf ("x=%a errf=%d  ", k, ir.first);
   printf (" errd=%d\n", ir.second);

}

  merr();

   return 0;
}
