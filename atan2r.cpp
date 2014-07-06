#include<cmath>
#include<utility>
#include<algorithm>
#include<iostream>
#include<string>


namespace fastmath_details {
  const double _2pi = (2.0 * 3.1415926535897932384626434);
  const float _2pif = float(_2pi);
  bool ataninited = false;
  float atanbuf_[257 * 2];
  double datanbuf_[513 * 2];

  namespace {
    // ====================================================================
    // arctan initialization
    // =====================================================================
    struct Initatan {
      Initatan() {
	if (ataninited)                   return;
	unsigned int ind;
	for (ind = 0; ind <= 256; ind++) {
	  double v = ind / 256.0;
	  double asinv = ::asin(v);
	  atanbuf_[ind * 2    ] = ::cos(asinv);
	  atanbuf_[ind * 2 + 1] = asinv;
	}
	for (ind = 0; ind <= 512; ind++) {
	  double v = ind / 512.0;
	  double asinv = ::asin(v);
	  datanbuf_[ind * 2    ] = ::cos(asinv);
	  datanbuf_[ind * 2 + 1] = asinv;
	}
	ataninited = true;
      }
    };
    static Initatan initAtan;
  }
}

// =====================================================================
// arctan, single-precision; returns phi and r
// =====================================================================
std::pair<float,float> atan2r(float y_, float x_) {
  using namespace fastmath_details;
  // assert(ataninited);
  float mag2 = x_ * x_ + y_ * y_;
  if(!(mag2 > 0))  {  return std::pair<float,float>(0.f,0.f); }   // degenerate case

  float r_ = std::sqrt(mag2);
  float rinv = 1.f/r_;
  unsigned int flags = 0;
  float x, y;
  float yp = 32768.f;
  if (y_ < 0 ) { flags |= 4; y_ = -y_; }
  if (x_ < 0 ) { flags |= 2; x_ = -x_; }
  if (y_ > x_) {
    flags |= 1;
    yp += x_ * rinv; x = rinv * y_; y = rinv * x_;
  }
  else {
    yp += y_ * rinv; x = rinv * x_; y = rinv * y_;
  }
  // r_ = rinv * mag2;
  int ind = (((int*)(&yp))[0] & 0x01FF) * 2;
  
  float* asbuf = (float*)(atanbuf_ + ind);
  float sv = yp - 32768.f;
  float cv = asbuf[0];
  float asv = asbuf[1];
  sv = y * cv - x * sv;    // delta sin value
  // double sv = y *(cv-x);
  // ____ compute arcsin directly
  float asvd = 6.f + sv * sv;   sv *= float(1.0f / 6.0f);
  float th = asv + asvd * sv;
  if (flags & 1) { th = (_2pif / 4.f) - th; }
  if (flags & 2) { th = (_2pif / 2.f) - th; }
  if (flags & 4) { th = -th; }
  return std::pair<float,float>(th,r_);
 
}

// =====================================================================
// arctan, double-precision; returns phi and r
// =====================================================================
std::pair<double, double> atan2r(double y_, double x_) {
  using namespace fastmath_details;
  // assert(ataninited);
  double mag2 = x_ * x_ + y_ * y_;
  if(!(mag2 > 0)) { return std::pair<double, double>(0.,0.); }   // degenerate case

  double r_ = sqrt(mag2);
  double rinv = 1./r_;
  unsigned int flags = 0;
  double x, y;
  const double _2p43 = 65536.0 * 65536.0 * 2048.0;
  double yp = _2p43;
  if (y_ < 0) { flags |= 4; y_ = -y_; }
  if (x_ < 0) { flags |= 2; x_ = -x_; }
  if (y_ > x_) {
    flags |= 1;
    yp += x_ * rinv; x = rinv * y_; y = rinv * x_;
  }
  else {
    yp += y_ * rinv; x = rinv * x_; y = rinv * y_;
  }
  // r_ = rinv * mag2;
  int ind = (((int*)(&yp))[0] & 0x03FF) * 2;  // 0 for little indian

  double* dasbuf = (double*)(datanbuf_ + ind);
  double sv = yp - _2p43; // index fraction
  double cv = dasbuf[0];
  double asv = dasbuf[1];
  sv = y * cv - x * sv;    // delta sin value
  // double sv = y *(cv-x);
  // ____ compute arcsin directly
  double asvd = 6 + sv * sv;   sv *= double(1.0 / 6.0);
  double th = asv + asvd * sv;
  if (flags & 1) { th = (_2pi / 4) - th; }
  if (flags & 2) { th = (_2pi / 2) - th; }
  if (flags & 4) { th = -th; }
  return std::pair<double, double>(th,r_);

}
 
#undef _2pi
#undef _2pif


namespace {

  template<typename T> 
  std::pair<T,T> stdatan2r(T x, T y) {
    return std::pair<T,T>(std::atan2(x,y),std::sqrt(x*x+y*y));
  }

  template<typename T>
  struct Stat {
    std::string name;
    size_t n;
    size_t npos;
    T bias;
    T ave;
    T rms;
    T amax;
    Stat( std::string in): name(in), n(0),npos(0), bias(0),ave(0),rms(0), amax(0){}
    void operator()(T x, T ref) {
      n++;
      if (x>ref) npos++;
      T d = (x-ref)/std::abs(ref);
      bias += d;
      ave +=std::abs(d);
      rms +=d*d;
      amax = std::max(amax,std::abs(d));
    }
    ~Stat() {
      std::cout << name << " "
		<< n << " " << npos << " " << bias/n << " " << ave/n 
		<< " " << (n*rms-bias*bias)/(n*(n-1))
		<< " " << amax << std::endl;
    }
  };

  volatile double dummy;

  template<typename T> 
  void sampleSquare() {
    Stat<T> stata("atan2");
    Stat<T> statr("r");
    T fac[8] = {-8, -5., -2., -1., 1.,2.,5.,8.};
    for (int k=0;k<100;k++)
      for (T x = 1e-15; x<1.1e+15; x *=10)
	for (T y = 1e-15; y<1.1e+15; y *=10) 
	  for (int i=0;i!=8; ++i)
	    for (int j=0;j!=8; ++j) {
	      T xx = x*fac[i];
	      T yy = y*fac[j];
	      std::pair<T,T> res = atan2r(xx,yy);
	      for (int l=0; l<i+j; ++l) dummy+=xx; // add a bit of random instruction
	      std::pair<T,T> ref = stdatan2r(xx,yy);
	      stata(res.first,ref.first);
	      statr(res.second,ref.second);
	    }
  }
  
}



#include<iostream>
int main() {

  {
  std::pair<double, double> res = atan2r(-3.,4.);
  std::cout << res.first << " " << res.second << std::endl;
  std::cout << atan2(-3.,4.)<< std::endl;
  }


  {
  std::pair<float, float> res = atan2r(3.f,-4.f);
  std::cout << res.first << " " << res.second << std::endl;
  std::cout << atan2f(3.f,-4.f)<< std::endl;
  
  }

  sampleSquare<float>();
  sampleSquare<double>();
  return 0;

}
