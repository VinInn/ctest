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
template<typename T>
class FastCircle {

public:

  FastCircle(){}
  FastCircle(T x1, T y1,
	     T x2, T y2,
	     T x3, T y3) { 
    compute(x1,y1,x2,y2,x3,y3);
  }

  void compute(T x1, T y1,
	       T x2, T y2,
	       T x3, T y3);
  
private:

  T m_xp;
  T m_yp;
  T m_c;
  T m_alpha;
  T m_beta;

};

#undef private

template<typename T>
void FastCircle<T>::compute(T x1, T y1,
			    T x2, T y2,
			    T x3, T y3) {
  bool flip = std::abs(x3-x1) > std::abs(y3-y1);
   
  auto x1p = x1-x2;
  auto y1p = y1-y2;
  auto d12 = x1p*x1p + y1p*y1p;
  auto x3p = x3-x2;
  auto y3p = y3-y2;
  auto d32 = x3p*x3p + y3p*y3p;

  if (flip) {
    std::swap(x1p,y1p);
    std::swap(x3p,y3p);
  }

  auto num = x1p*y3p-y1p*x3p;  // num also gives correct sign for CT
  auto det = d12*y3p-d32*y1p;
  if( std::abs(det)==0 ) {
    // and why we flip????
  }
  auto ct  = num/det;
  auto sn  = det>0 ? 1 : -1;  
  auto st2 = (d12*x3p-d32*x1p)/det;
  auto seq = 1. +st2*st2;
  auto al2 = sn/sqrt(seq);
  auto be2 = -st2*al2;
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
template<typename T>
void verify(T x1, T y1,
	    T x2, T y2,
	    T x3, T y3) {
  FastCircle<T> c;
  c.compute(x1,y1,x2,y2,x3,y3);
  std::cout << c.m_xp <<"," << c.m_yp
	    << " ; " << c.m_c
	    << " " << c.m_alpha <<"," << c.m_beta
	    << std::endl;
}

template<typename T>
bool equal(T a, T b) {
  //  return float(a-b)==0;
  return std::abs(float(a-b))<1.e-10;
}

template<typename T>
bool equal(FastCircle<T> const & a, FastCircle<T> const & b) {
  return equal(a.m_xp,b.m_xp) &&
    equal(a.m_yp,b.m_yp) &&
    equal(a.m_c,b.m_c) &&
    equal(a.m_alpha,b.m_xp) &&
    equal(a.m_xp,b.m_xp);

} 

template<typename T>
void go() {

  T one = 1.0, zero=0., c4 = cos(4.), s4 = sin(4.), c1 = cos(0.1), s1 = sin(0.1);
    
    verify(one,zero,c4,s4,zero,one);
  verify(one,zero,one,zero,c4,s4);
  verify(one,zero,c4,s4,c1,s1);
  verify(one,zero,c1,s1,c4,s4);

  for (T phi=0; phi<6.28; phi+=0.05*M_PI) {
    T x1 = cos(phi), y1 = sin(phi);
    {
      T x2 = cos(phi+0.1), y2 = sin(phi+0.1);
      T x3 = cos(phi+0.2), y3 = sin(phi+0.2);
      verify (x1,y1,x2,y2,x3,y3);
      verify (x1,y1,x3,y3,x2,y2);
    }
    {
      T x2 = cos(phi-0.1), y2 = sin(phi-0.1);
      T x3 = cos(phi-0.2), y3 = sin(phi-0.2);
      verify (x1,y1,x2,y2,x3,y3);
    }
  }

  T phi=0;
  T x1 = cos(phi), y1 = sin(phi);
  T d=0.01;
  T x2 = cos(phi+d), y2 = sin(phi+d);
  T x3 = cos(phi+0.2), y3 = sin(phi+0.2);
  FastCircle<T> ca(x1,y1,x2,y2,x3,y3);
  FastCircle<T> cb;
  do {
    d*=T(0.1);
    x2 = cos(phi+d), y2 = sin(phi+d);
    cb.compute(x1,y1,x2,y2,x3,y3);
  } while(equal(ca.m_c,cb.m_c));
  std::cout << d << std::endl;
  verify(x1,y1,x2,y2,x3,y3);
	 
  d=0.1; long long n=0;
  do {
    ++n;
    d *= T(0.1);
    ca.compute(d,zero,zero,one,zero,-one);
  }while (ca.m_c!=0);
  std::cout << n << "  " << d << std::endl;
  verify(zero,one,d,zero,zero,-one);
  verify(zero,one,std::numeric_limits<T>::min(),zero,zero,-one);
  verify(zero,one,-std::numeric_limits<T>::min(),zero,zero,-one);

  verify(zero,zero,one,one,T(2.),T(2.));

}

int main() {

  std::cout << std::numeric_limits<double>::max() << std::endl;
  std::cout << std::numeric_limits<double>::min() << std::endl;
  std::cout << std::abs(float(std::numeric_limits<double>::min())) << std::endl;
  std::cout << std::numeric_limits<float>::max() << std::endl;
  std::cout << std::numeric_limits<float>::min() << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  go<double>();
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  go<float>();

  return 0;
};

