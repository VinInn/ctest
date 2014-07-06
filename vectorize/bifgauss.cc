#include<cmath>
double x[1024], y[1024];


inline double oriOne(double x, double m, double sl, double sr) {
  double arg = x-m;
  double coef=0;
  if (arg < coef) {
    if (std::abs(sl)>1e-30) 
      {coef = -(0.5)/(sl*sl); }
  } else {
    if (std::abs(sr)>1e-30) 
      {coef = -(0.5)/(sr*sr); }
 }


  return coef*arg*arg;

}

void fooOri(double m, double sl, double sr) {
  // #pragma omp simd 
   for (int i=0; i<1024; ++i) y[i] = oriOne(x[i],m,sl,sr);

}


inline double fooOne(double x, double m, double sl, double sr) {
  double arg = x-m;
  double coef = (arg<0) ? -0.5/(sl*sl) : -0.5/(sr*sr);

  return coef*arg*arg;

}

void foo(double m, double sl, double sr) {
   for (int i=0; i!=1024; ++i) y[i] = fooOne(x[i],m,sl,sr);

}




inline double barOne(double x, double m, double sl, double sr) {
  double arg = x-m;
  double s = (arg<0) ? sl : sr;
  double r = arg/s;
  return -0.5*r*r;

}

void bar(double m, double sl, double sr) {
   for (int i=0; i!=1024; ++i) y[i] = barOne(x[i],m,sl,sr);
}


