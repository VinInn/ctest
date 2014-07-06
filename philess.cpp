#include <iostream>
#include <cmath>

class PhiLess {
public:
  bool operator()(float a , float b) const
  {
    const float pi = 3.141592653592;
    float diff = std::fmod(b - a, 2*pi);
    if ( diff < 0) diff += 2*pi;
    return diff < pi;
  }
};

inline bool phiLess( float phi1, float phi2) {
  float diff = phi2-phi1; 
  if ( diff < 0) diff += 2*M_PI;
  return diff < M_PI;
}


//PROXIM(B,A)=B+C1*ANINT(C2*(A-B)) , with C1=2*PI and C2 = 1/C1
/*
double deltaPhi(float a, float b) {
  const float pi = 3.141592653592;
  double diff =  std::fmod(a-b, 2*pi);
  if ( diff < 0) diff += pi;
  return diff;
}
*/

/*  bring a in range b-pi,b+pi
 */
double phiInRange(float a, float b) {
  const float pi = 3.141592653592;
  return b + std::fmod(a-b, 2*pi);
}

void test(float a, float b) {
  PhiLess less;

  if (less(a,b) ) 
    std::cout << a << "<"  << b << std::endl;
  else std::cout << a << ">" <<  b << std::endl;

}

void test2(float a, float b) {

  if (phiLess(a,b) ) 
    std::cout << a << "<"  << b << std::endl;
  else std::cout << a << ">" <<  b << std::endl;
}

inline double deltaPhi(double phi1, double phi2) { 
  double dphi = phi2-phi1; 
  if ( dphi > M_PI ) {
    dphi -= 2.0*M_PI;
  } else if ( dphi <= -M_PI ) {
    dphi += 2.0*M_PI;
  }
  return dphi;
}


void tdp(float a, float b) {
  std:: cout << a << " "  
	     << b << " " 
	     << deltaPhi(a,b) 
	     << std::endl;

}

int main() {

  const float pi = 3.141592653592;
  float a = 2*pi/3-0.001;
  float b = 4*pi/3-0.001;
  float c = 6*pi/3-0.001;
  float d = -0.001;

  tdp(a,a);
  tdp(a,-a);
  tdp(-a,a);
  tdp(a,b);
  tdp(b,a);
  tdp(b,c);
  tdp(c,b);
  tdp(a,c);
  tdp(c,a);
  tdp(c,-c);
  tdp(-c,c);
  tdp(a,d);
  tdp(d,a);
  tdp(d,c);
  tdp(d,-c);
  tdp(-d,c);
  std::cout << std::endl;

  test(a,-a);
  test(a,b);
  test(b,c);
  test(c,a);
  test(c,-c);

  test(b,a);
  test(c,b);
  test(a,c);

  std::cout << std::endl;

  test(-a,-b);
  test(-b,-c);
  test(-c,-a);

  test(-b,-a);
  test(-c,-b);
  test(-a,-c);
  std::cout << std::endl;

  test2(a,-a);
  test2(a,b);
  test2(b,c);
  test2(c,a);
  test2(c,-c);

  test2(b,a);
  test2(c,b);
  test2(a,c);

  std::cout << std::endl;

  test2(-a,-b);
  test2(-b,-c);
  test2(-c,-a);

  test2(-b,-a);
  test2(-c,-b);
  test2(-a,-c);


  std::cout << "\n"
	    << std::atan2(0.,1.) << " " 
	    << std::atan2(1.,0.) << " " 
	    << std::atan2(.01,-1.) << " " 
	    << std::atan2(-.01,-1.) << " " 
	    << std::atan2(-1.,0.) << std::endl; 
    

  std::cout << "\n"
	    << a << " " << std::fmod(a, pi) << "\n"
	    << -a << " " << std::fmod(-a, pi) << "\n"
	    << a+4.f*pi << " " << std::fmod(a+4.f*pi, pi) << "\n"
	    << a-4.f*pi << " " << std::fmod(a-4.f*pi, pi) << "\n"
	    << c << " " << std::fmod(c, pi) << "\n"
	    << -c << " " << std::fmod(-c, pi) << std::endl; 
  std::cout << "\n"
	    << a << " " << phiInRange(a, pi) << "\n"
	    << -a << " " << phiInRange(-a, pi) << "\n"
	    << a+4.f*pi << " " << phiInRange(a+4.f*pi, pi) << "\n"
	    << a-4.f*pi << " " << phiInRange(a-4.f*pi, pi) << "\n"
	    << c << " " << phiInRange(c, pi) << "\n"
	    << -c << " " << phiInRange(-c, pi) << std::endl; 

  std::cout << "\n"
	    << a << " " << deltaPhi(a, pi) << "\n"
	    << -a << " " << deltaPhi(-a, pi) << "\n"
	    << pi-0.01f << " " << deltaPhi(pi-0.01f, pi) << "\n"
	    << pi+0.01f << " " << deltaPhi(pi+0.01f, pi) << "\n"
	    << -pi+0.01f << " " << deltaPhi(-pi+0.01f, pi) << "\n"
	    << a+4.f*pi << " " << deltaPhi(a+4.f*pi, pi) << "\n"
	    << a-4.f*pi << " " << deltaPhi(a-4.f*pi, pi) << "\n"
	    << c << " " << deltaPhi(c, pi) << "\n"
	    << -c << " " << deltaPhi(-c, pi) << std::endl; 

  std::cout << "\n"
	    << std::fmod(a, -pi) << "\n"
	    << std::fmod(a+4.f*pi, -pi) << "\n"
	    << std::fmod(c, -pi) << "\n"
	    << std::fmod(-c, -pi) << std::endl; 

  std::cout << "\n"
	    << std::fmod(a, 2*pi) << "\n"
	    << std::fmod(a+4.f*pi, 2*pi) << "\n"
	    << std::fmod(c, 2*pi) << "\n"
	    << std::fmod(-c, 2*pi) << std::endl; 



}



