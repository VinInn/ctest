#include<cmath>
static float DP1 = 0.78515625;
static float DP2 = 2.4187564849853515625e-4;
static float DP3 = 3.77489497744594108e-8;
static float lossth = 8192.;
static float T24M1 = 16777215.;
static float FOPI = 1.27323954473516;
static float PIO4F = 0.7853981633974483096;
static float PIF = 3.141592653589793238;
static float PIO2F = 1.5707963267948966192;

static double dDP1 = 0.78515625;
static double dDP2 = 2.4187564849853515625e-4;
static double dDP3 = 3.77489497744594108e-8;
static double dFOPI = 1.27323954473516;



#include<iostream>

inline void prec(float x) {

  int j = FOPI * x; /* integer part of x/PIO4 */
  float y = j;
  if( j & 1 ) y+=1;

  int dj = dFOPI * x; /* integer part of x/PIO4 */
  float yd = dj;
  if( j & 1 ) yd+=1;

  float x1 = x - y * PIO4F;
  float x2 = ((x - y * DP1) - y * DP2) - y * DP3;
  float x3 = ((x - yd * dDP1) - yd * dDP2) - yd * dDP3;

  float z = x2*x2;
  float y1 = (((-1.9515295891E-4f * z
		+ 8.3321608736E-3f) * z
	       - 1.6666654611E-1f) * z * x2)
    + x2;

  float y2 = ((  2.443315711809948E-005f * z
	  - 1.388731625493765E-003f) * z
       + 4.166664568298827E-002f) * z * z;
  y2 -= 0.5 * z;
  y2 += 1.0;

  printf("%e %e %e %e %e %e %e %e\n",x,x1,x2,x3,y1,y2,cos(x),sin(x));

}

int main() {

  prec(2.75f);
  prec(10.32f);
  prec(10.32001f);
  prec(10.33f);
  prec(lossth);
  prec(lossth-0.1f);
  prec(lossth+0.1f);
  prec(lossth+10.f);
  prec(lossth+100.f);
  prec(10.f*lossth);
  prec(100.f*lossth);
  
  return 0;
}
