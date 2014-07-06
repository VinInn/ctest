#include<cmath>

class ThirdHitPredictionFromInvParabola {

public:

  double doit(double r, int c, double i1, double i2) const;

  inline double coeffA(double impactParameter, int charge) const;
  inline double coeffB(double impactParameter, int charge) const;
  inline double ipFromCurvature(double  curvature, int charge) const;
  
  
  inline void findPointAtCurve(double radius, int charge, double ip, double & u, double & v) const;
  
  double u1u2, overDu, pv, dv, su;
  
};

double  ThirdHitPredictionFromInvParabola::
    coeffA(double impactParameter, int charge) const
{
  return -charge*pv*overDu - u1u2*impactParameter;
}

double ThirdHitPredictionFromInvParabola::
    coeffB(double impactParameter,int charge) const
{
  return charge*dv*overDu - su*impactParameter;
}

double ThirdHitPredictionFromInvParabola::
    ipFromCurvature(double curvature, int charge) const 
{
  double overU1u2 = 1./u1u2;
  double inInf = -charge*pv*overDu*overU1u2;
  return inInf-curvature*overU1u2*0.5;
}

void ThirdHitPredictionFromInvParabola::findPointAtCurve(double r, int c, double ip, 
							 double & u, double & v) const
{
  //
  // assume u=(1-alpha^2/2)/r v=alpha/r
  // solve qudratic equation neglecting aplha^4 term
  //
  double A = coeffA(ip,c);
  double B = coeffB(ip,c);

  // double overR = 1./r;
  double ipOverR = ip/r; // *overR;

  double delta = 1-4*(0.5*B+ipOverR)*(-B+A*r-ipOverR);
  double sqrtdelta = (delta > 0) ? std::sqrt(delta) : 0.;
  //  double sqrtdelta = std::sqrt(delta);
  double alpha = (c>0)?  (-c+sqrtdelta)/(B+2*ipOverR) :  (-c-sqrtdelta)/(B+2*ipOverR);

  v = alpha;  // *overR
  double d2 = 1. - v*v;  // overR*overR - v*v
  u = (d2 > 0) ? std::sqrt(d2) : 0.;
  // u = std::sqrt(d2);

  // u,v not rotated! not multiplied by 1/r
}


double ThirdHitPredictionFromInvParabola::doit(double r, int c, double i1, double i2) const {
  double ip[2]={i1,i2};
  double u[2], v[2];
  for (int i=0; i!=2; ++i)
    findPointAtCurve(r,c, ip[i],u[i],v[i]);


  return (u[0]+u[1])/(v[0]+v[1]);
}
