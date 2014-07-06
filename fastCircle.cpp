#include<cmath>
#include<algorithm>
#include<numeric>


/**
1) circle is parameterized as:                                              |
|    C*[(X-Xp)**2+(Y-Yp)**2] - 2*alpha*(X-Xp) - 2*beta*(Y-Yp) = 0             |
|    Xp,Yp is a point on the track (Yp is at the center of the chamber);      |
|    C = 1/r0 is the curvature  ( sign of C is charge of particle );          |
|    alpha & beta are the direction cosines of the radial vector at Xp,Yp     |
|    i.e.  alpha = C*(X0-Xp),                                                 |
|          beta  = C*(Y0-Yp),                                                 |
|    where center of circle is at X0,Y0.                                      |
|    Alpha > 0                                                                |
|    Slope dy/dx of tangent at Xp,Yp is -alpha/beta.                          |
| 2) the z dimension of the helix is parameterized by gamma = dZ/dSperp       |
|    this is also the tangent of the pitch angle of the helix.                |
|    with this parameterization, (alpha,beta,gamma) rotate like a vector.     |
| 3) For tracks going inward at (Xp,Yp), C, alpha, beta, and gamma change sign|
|
*/

#define private public
class FastCircle {

public:

  FastCircle(){}
  FastCircle(double x1, double y1,
	     double x2, double y2,
	     double x3, double y3) { 
    compute(x1,y1,x2,y2,x3,y3);
  }

  void compute(double x1, double y1,
	       double x2, double y2,
	       double x3, double y3);

private:

  double m_xp;
  double m_yp;
  double m_c;
  double m_alpha;
  double m_beta;

};

#undef private

void FastCircle::compute(double x1, double y1,
			 double x2, double y2,
			 double x3, double y3) {
  bool flip = fabs(x3-x1) > fabs(y3-y1);
   
  double x1p = x1-x2;
  double y1p = y1-y2;
  double d12 = x1p*x1p + y1p*y1p;
  double x3p = x3-x2;
  double y3p = y3-y2;
  double d32 = x3p*x3p + y3p*y3p;

  if (flip) {
    std::swap(x1p,y1p);
    std::swap(x3p,y3p);
  }

  double num = x1p*y3p-y1p*x3p;  // num also gives correct sign for CT
  double det = d12*y3p-d32*y1p;
  if( std::fabs(det)==0 ) {
    // and why we flip????
  }
  double ct  = num/det;
  double sn  = det>0 ? 1 : -1;  
  double st2 = (d12*x3p-d32*x1p)/det;
  double seq = 1. +st2*st2;
  double al2 = sn/sqrt(seq);
  double be2 = -st2*al2;
  ct *= 2.*al2;
  
  if (flip) {
    std::swap(x1p,y1p);
    std::swap(al2,be2);
    al2 = -al2;
    be2 = -be2;
    ct = -ct;
  }
  
  m_xp = x1;
  m_yp = y1;
  m_c= ct;
  m_alpha = al2 - ct*x1p;
  m_beta = be2 - ct*y1p;
  
}

#include<iostream>
void verify(double x1, double y1,
	    double x2, double y2,
	    double x3, double y3) {
  FastCircle c;
  c.compute(x1,y1,x2,y2,x3,y3);
  std::cout << c.m_xp <<"," << c.m_yp
	    << " ; " << c.m_c
	    << " " << c.m_alpha <<"," << c.m_beta
	    << std::endl;
}


bool equal(double a, double b) {
  //  return float(a-b)==0;
  return abs(float(a-b))<1.e-15;
}

bool equal(FastCircle const & a, FastCircle const & b) {
  return equal(a.m_xp,b.m_xp) &&
    equal(a.m_yp,b.m_yp) &&
    equal(a.m_c,b.m_c) &&
    equal(a.m_alpha,b.m_xp) &&
    equal(a.m_xp,b.m_xp);

} 

int main() {

  std::cout << std::numeric_limits<double>::max() << std::endl;
  std::cout << std::numeric_limits<double>::min() << std::endl;
  std::cout << fabs(float(std::numeric_limits<double>::min())) << std::endl;
  std::cout << std::numeric_limits<float>::max() << std::endl;
  std::cout << std::numeric_limits<float>::min() << std::endl;

  verify(1.,0.,cos(4.),sin(4),0.,1.);
  verify(1.,0.,0.,1.,cos(4.),sin(4));
  verify(1.,0.,cos(4.),sin(4.),cos(0.1),sin(0.1));
  verify(1.,0.,cos(0.1),sin(0.1),cos(4.),sin(4.));

  for (double phi=0; phi<6.28; phi+=0.05*M_PI) {
    double x1 = cos(phi), y1 = sin(phi);
    {
      double x2 = cos(phi+0.1), y2 = sin(phi+0.1);
      double x3 = cos(phi+0.2), y3 = sin(phi+0.2);
      verify (x1,y1,x2,y2,x3,y3);
      verify (x1,y1,x3,y3,x2,y2);
    }
    {
      double x2 = cos(phi-0.1), y2 = sin(phi-0.1);
      double x3 = cos(phi-0.2), y3 = sin(phi-0.2);
      verify (x1,y1,x2,y2,x3,y3);
    }
  }

  double phi=0;
  double x1 = cos(phi), y1 = sin(phi);
  double d=0.01;
  double x2 = cos(phi+d), y2 = sin(phi+d);
  double x3 = cos(phi+0.2), y3 = sin(phi+0.2);
  FastCircle c1(x1,y1,x2,y2,x3,y3);
  FastCircle c2;
  do {
    d*=0.1;
    x2 = cos(phi+d), y2 = sin(phi+d);
    c2.compute(x1,y1,x2,y2,x3,y3);
  } while(equal(c1.m_c,c2.m_c));
  std::cout << d << std::endl;
  verify(x1,y1,x2,y2,x3,y3);
	 
  d=0.1;
  do {
    d *= 0.1;
    c1.compute(0.,1.,d,0.,0.,-1.);
  }while (c1.m_c!=0.);
  std::cout << d << std::endl;
  verify(0.,1.,d,0.,0.,-1.);
  verify(0.,1.,std::numeric_limits<double>::min(),0.,0.,-1.);
  verify(0.,1.,-std::numeric_limits<double>::min(),0.,0.,-1.);

  verify(0.,0.,1.,1.,2.,2.);

  return 0;
};

