#include"exp.h"
struct Argus3 {
 double x[1024], y[1024];


 inline
 double one(double m, double m0, double c) {
   double t= m/m0;
   double u= 1. - t*t;
   // return (t >= 1.) ? 0. :  m*std::sqrt(u)*vdt::fast_exp(c*u) ;
   return m*std::sqrt(u)*vdt::fast_exp(c*u) ;
  }


  virtual void compute(double m, double m0, double c);

};


void Argus3::compute(double m, double m0, double c) {
    for (int i=0; i!=1024; ++i) y[i] =  one(x[i],m0,c);
  }

