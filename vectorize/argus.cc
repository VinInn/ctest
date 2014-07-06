#include<cmath>
double x[1024], y[1024];


inline
double fooOne(double m, double m0, double c) {

  double t= m/m0;

  double u= 1. - t*t;
    
  return (t >= 1.) ? 0. :  m*std::sqrt(u)*(c*u) ;


}


void foo(double m, double m0, double c) {
   for (int i=0; i!=1024; ++i) y[i] = fooOne(x[i],m0,c);

}


