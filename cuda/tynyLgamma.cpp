#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

double ngamma(double x) {
    double ax = fabs(x);
    // uint hax = AS_UINT2(ax).hi;
    struct ID { double d; unsigned long long u;};
    ID id; id.d = ax; uint32_t hax = (id.u >> 32);
    double ret=0;

if (hax < 0x43300000) { // x > -0x1.0p+52
        double t = sinpi(x);
        double negadj = log(M_PI/fabs(t * x)));
        ret = negadj - ret;
        // bool z = BUILTIN_FRACTION_F64(x) == 0.0;
        // ret = z ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        s = t < 0.0 ? -1 : 1;
        // s = z ? 0 : s;
    } else std::cout << x << "too large" << std::endl;
return ret;
}


double gamma(double x) {
    const double z1  = -0x1.2788cfc6fb619p-1;
    const double z2  =  0x1.a51a6625307d3p-1;
    const double z3  = -0x1.9a4d55beab2d7p-2;
    const double z4  =  0x1.151322ac7d848p-2;
    const double z5  = -0x1.a8b9c17aa6149p-3;

    double ax = fabs(x);
    // uint hax = AS_UINT2(ax).hi;
    struct ID { double d; unsigned long long u;};
    ID id; id.d = ax; uint32_t hax = (id.u >> 32);
    double ret=0;

#define MATH_MAD(a,b,c) fma(a,b,c)

    if (hax < 0x3f700000) {
        // ax < 0x1.0p-8
        ret = MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, z5, z4), z3), z2), z1),
                       -log(ax));
    } else std::cout << x << "too large" << std::endl;

   return ret;
}


#include<cstdio>
int main() {

  double x= -0x1.5efad5491a79bp-1022;
  std::cout  <<" lgamma for " << x << " = " << gamma(x) << " " << ngamma(x) << ' ' << log(fabs(x)) << std::endl;

  printf("x log(fabs(x)) %a %a\n",x,log(fabs(x)));

}
