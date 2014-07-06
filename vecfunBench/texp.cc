#include<cmath>
typedef union {
  double d;
  int i[2];
  long long ll;
  unsigned short s[4];
} ieee754;

static const double MAXLOG =  7.08396418532264106224E2;     /* log 2**1022 */
static const double MINLOG = -7.08396418532264106224E2;     /* log 2**-1022 */
static const double LOG2E  =  1.4426950408889634073599;     /* 1/log(2) */
static const double DINFINITE = 1.79769313486231570815E308;
static const double C1 = 6.93145751953125E-1;
static const double C2 = 1.42860682030941723212E-6;


double values[1024];
void vecexp_cephes()
{
    int i;
    for (i = 0;i< 1024;++i) {
        int n;
        double x = values[i];
        double a, xx, px, qx;
        ieee754 u;

        /* n = round(x / log 2) */
        // a = LOG2E * x + 0.5;
        // n = (int)a;
        // n -= (a< 0);

        /* x -= n * log2 */
        // px = (double)n;

        px = std::floor( LOG2E * x + 0.5 );
//        px = int( LOG2E * x + 0.5 );

        /* x -= n * log2 */
        n = px;
  
        x -= px * C1;
        x -= px * C2;
        xx = x * x;

        /* px = x * P(x**2). */
        px = 1.26177193074810590878E-4;
        px *= xx;
        px += 3.02994407707441961300E-2;
        px *= xx;
        px += 9.99999999999999999910E-1;
        px *= x;

        /* Evaluate Q(x**2). */
        qx = 3.00198505138664455042E-6;
        qx *= xx;
        qx += 2.52448340349684104192E-3;
        qx *= xx;
        qx += 2.27265548208155028766E-1;
        qx *= xx;
        qx += 2.00000000000000000009E0;

        /* e**x = 1 + 2x P(x**2)/( Q(x**2) - P(x**2) ) */
        x = px / (qx - px);
        x = 1.0 + 2.0 * x;

        /* Build 2^n in double. */
        u.d = 0;
        n += 1023;
        // u.s[3] = (unsigned short)((n<< 4) & 0x7FF0);
        // u.i[1] =  n <<20;
        u.ll =  (long long)(n) <<52;

        values[i] = x * u.d;
    }
}
