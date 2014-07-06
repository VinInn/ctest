#include<array>
#include<cmath>

#include "approx_exp.h"

inline
int diff(float a, float b) {
  approx_math::binary32 ba(a);
  approx_math::binary32 bb(b);
  return ba.i32 - bb.i32;

}


inline
int bits(int a) {
  unsigned int aa = abs(a);
  int b=0; if (a==0) return 0;
  while ( (aa/=2) > 0 )  ++b;
  return (a>0) ? b : -b;

}


template<typename T>
struct BCylParam {
  //  constexpr BCylParam(std::initializer_list<T> init) :
  template<typename ... Args>
  constexpr BCylParam(Args... init) :
    prm{std::forward<Args>(init)...},
    //prm(std::forward<Args>(init)...), 
    ap2(4*prm[0]*prm[0]/(prm[1]*prm[1])), 
    hb0(0.5*prm[2]*std::sqrt(1.0+ap2)),
    hlova(1/std::sqrt(ap2)),
    ainv(2*hlova/prm[1]),
    coeff(1/(prm[8]*prm[8])){}

  T prm[9];
  T ap2, hb0, hlova, ainv,coeff;


};


//constexpr
BCylParam<double>  dpar1{4.90541,17.8768,2.02355,0.0210538,0.000321885,2.37511,0.00326725,2.07656,1.71879}; // 2.0T-2G
BCylParam<double>  dpar2{4.41982,15.7732,3.02621,0.0197814,0.000515759,2.43385,0.00584258,2.11333,1.76079}; // 3.0T-2G
BCylParam<double>  dpar3{4.30161,15.2586,3.51926,0.0183494,0.000606773,2.45110,0.00709986,2.12161,1.77038}; // 3.5T-2G
BCylParam<double>  dpar4{4.24326,15.0201,3.81492,0.0178712,0.000656527,2.45818,0.00778695,2.12500,1.77436}; // 3.8T-2G
BCylParam<double>  dpar5{4.21136,14.8824,4.01683,0.0175932,0.000695541,2.45311,0.00813447,2.11688,1.76076}; // 4.0T-2G

BCylParam<float>  fpar1{4.90541,17.8768,2.02355,0.0210538,0.000321885,2.37511,0.00326725,2.07656,1.71879}; // 2.0T-2G
BCylParam<float>  fpar2{4.41982,15.7732,3.02621,0.0197814,0.000515759,2.43385,0.00584258,2.11333,1.76079}; // 3.0T-2G
BCylParam<float>  fpar3{4.30161,15.2586,3.51926,0.0183494,0.000606773,2.45110,0.00709986,2.12161,1.77038}; // 3.5T-2G
BCylParam<float>  fpar4{4.24326,15.0201,3.81492,0.0178712,0.000656527,2.45818,0.00778695,2.12500,1.77436}; // 3.8T-2G
BCylParam<float>  fpar5{4.21136,14.8824,4.01683,0.0175932,0.000695541,2.45311,0.00813447,2.11688,1.76076}; // 4.0T-2G

namespace bcylDetails{

  template<typename T>
  inline void ffunkti(T u, T * __restrict__ ff) __attribute__((always_inline)) __attribute__ ((pure));
 
  template<typename T>
  inline void ffunkti(T u, T * __restrict__ ff) {
    // Function and its 3 derivatives
    T a,b,a2,u2;
    u2=u*u; 
    a=T(1)/(T(1)+u2);
    a2=-T(3)*a*a;
    b=std::sqrt(a);
    ff[0]=u*b;
    ff[1]=a*b;
    ff[2]=a2*ff[0];
    ff[3]=a2*ff[1]*(T(1)-4*u2);
  }

  double myExp(double x) { return std::exp(x);}
  float myExp(float x) { return unsafe_expf<3>(x);}

}

template<typename T>
class BCycl  {
public:
  BCycl(BCylParam<T> const & ipar) : pars(ipar) {} 

  void operator()(T r2, T z, T& Br, T& Bz) const {
    compute(r2,z,Br,Bz);
  }


  // in meters and T (br shall be multiplied by r...)
  void compute(T r2, T z, T& Br, T& Bz) const {
    using namespace  bcylDetails;
    //  if (r<1.15&&fabs(z)<2.8) // NOTE: check omitted, is done already by the wrapper! (NA)
    z-=pars.prm[3];                    // max Bz point is shifted in z
    T az=std::abs(z);
    T zainv=z*pars.ainv;
    T u=pars.hlova-zainv;
    T v=pars.hlova+zainv;
    T fu[4],gv[4];
    ffunkti(u,fu);
    ffunkti(v,gv);
    T rat=T(0.5)*pars.ainv;
    T rat2=rat*rat*r2;
    Br=pars.hb0*rat*(fu[1]-gv[1]-(fu[3]-gv[3])*rat2*T(0.5));
    Bz=pars.hb0*(fu[0]+gv[0]-(fu[2]+gv[2])*rat2);

    T corBr= pars.prm[4]*z*(az-pars.prm[5])*(az-pars.prm[5]);
    T corBz=-pars.prm[6]*(
			  myExp(-(z-pars.prm[7])*(z-pars.prm[7])*pars.coeff) +
			  myExp(-(z+pars.prm[7])*(z+pars.prm[7])*pars.coeff)
			  ); // double Gaussian
    Br+=corBr;
    Bz+=corBz;
  }

private:
  BCylParam<T> pars;

};

namespace testVect {
  constexpr int NN=1024;
  float r[NN],z[NN],br[NN],bz[NN];
  void go() {
    BCycl<float> fbc(fpar4);
    for (int i=0; i!=NN; ++i)
      fbc(r[i],z[i],br[i],bz[i]);
  }
  void g6() {
    BCycl<float> fbc(fpar4);
    for (int i=0; i!=8; ++i)
      fbc(r[i],z[i],br[i],bz[i]);
  }
}


#include<iostream>
#include<algorithm>
int main() {
  std::cout << "double ";
  for (auto p : dpar4.prm)
    std::cout << p << " ";
  std::cout << ", " << dpar4.ap2 << " " << dpar4.coeff << std::endl;
  std::cout << "float  ";
  for (auto p : fpar4.prm)
    std::cout << p << " ";
  std::cout << ", " << fpar4.ap2 << " " << fpar4.coeff << std::endl;

  BCycl<double> dbc(dpar4);
  BCycl<float> fbc(fpar4);

  {
    double dbr, dbz;
    float fbr,fbz;
    dbc(0,0,dbr,dbz);
    fbc(0,0,fbr,fbz);
    std::cout << "at 0 d " << dbr << " " << dbz << std::endl;
    std::cout << "at 0 d " << fbr << " " << fbz << std::endl;
  }

  int diffR=0;
  int diffZ=0;
  double mdbr, mdbz;
  float mfbr, mfbz;
 
  for (float z=-2.8; z<2.8;z+=0.1)
    for (float r=0; r<1.15;r+=0.1) {
      double dbr, dbz;
      float fbr,fbz;
      dbc(r*r,z,dbr,dbz);
      fbc(r*r,z,fbr,fbz);
      int dr = diff(fbr,float(dbr));
      int dz = diff(fbz,float(dbz));
      if (dr> diffR) {
	diffR=dr; mfbr = fbr; mdbr=dbr;
      } 
      if (dz> diffZ) {
	diffZ=dz; mfbz = fbz; mdbz=dbz;
      } 
    }
  std::cout << "max diff r,z " << diffR <<" " << diffZ << std::endl;
  std::cout << "max diff r,z in bits " << bits(diffR) <<" " << bits(diffZ) << std::endl;
  std::cout << "Br " << mdbr << " " << mfbr << std::endl;
  std::cout << "Bz " << mdbz << " " << mfbz << std::endl;



}


