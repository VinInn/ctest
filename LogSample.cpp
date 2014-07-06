#include<cmath>
#include<vector>


struct LogScale {
  
  template<typename F>
  LogScale(float imin, float imax, F f) {
    frexpf(imin,&nMin); --nMin;
    frexpf(imax,&nMax);
    values.resize(size(),0.);
    for (int n=nMin; n<=nMax; ++n)
      values[n-nMin] = f(ldexpf(1.,n));
  }
  
  int size() const { return nMax-nMin +1;}

  static std::pair<int, float> bin(float x) {
    const float l2 = 5*log10(2.);
    const float p1 = l2;
    const float p2 = -l2/2.;
    const float p3 = l2/3.;

    std::pair<int, float> res;
    float f = frexpf(x,&res.first)-1.f;
    res.second = f*(p1+f*(p2+p3*f));
    return res;
  }

  float value(float x) const {
    std::pair<int, float> res = bin(x);
    return  values[res.first-nMin] + res.second*(values[res.first-nMin] - values[res.first-nMin-1]);
  }


  int nMin;
  int nMax;
  
  std::vector<float> values;

};

float compute(float x,  LogScale const & s) {
  return s.value(x);
}

#include<cstdlib>
#include<cstdio>
#include<iostream>
#include<iomanip>

void look(float x) {
  int e;
  float f = ::frexpf(x,&e);
  std::cout << x << " exp " << e << " res " << f << std::endl;

  union {
    float val;
    int x;
  } tmp;
  tmp.val = x;
  const int log_2 = ((tmp.x >> 23) & 255) - 127;//exponent
  tmp.x &= 0x7FFFFF; //mantissa
  //     0xFF800000;
  std::cout << "exp " << log_2 << " mant as int "  << std::hex << tmp.x 
	    << " mant as float " <<  std::dec << (tmp.x|0x800000)*::pow(2.,-23) << std::endl;
  // approx
  int n = 10;
  int mask = 0x1 << (23-n-1);
  tmp.val = x;
  int rounding = (tmp.x & mask) ? 1 : 0;
  int m = (tmp.x & 0x7FFFFF) >> (23-n); 
  tmp.x &= 0xFF800000; tmp.x |=(m<<23-n);
  std::cout << "approx " << tmp.val << " " << std::hex <<  (tmp.x & 0x7FFFFF)  <<  std::dec << " " << m;
  tmp.val = x;
  int m2 = (tmp.x & 0x7FFFFF) >> (23-n-1); 
  tmp.x &= 0xFF800000; tmp.x |=(m2<<23-n-1);
  std::cout << " " << tmp.val <<  " "  << std::hex << (tmp.x & 0x7FFFFF) <<  std::dec  << " " << m2 << std::endl;
  //  if ( m2 & 0x1 ) {
  m+=rounding; tmp.val = x;  tmp.x &= 0xFF800000; tmp.x |=(m<<23-n);
  std::cout << "better " << std::hex << tmp.val <<  " " << (tmp.x & 0x7FFFFF) <<  std::dec << " " << m << std::endl;
  // }
}

int main() {

  std::cout << "have a look inside a float" << std::endl;
  look(0.f);
  look(0.99999f);
  look(1.f);
  look(2.f);
  look(3.f);
  look(std::sqrt(3.3));
  look(std::sqrt(333444.e14));
  look(std::sqrt(333444.e-14));




  LogScale scale(10.e-2,10.e5,::log10f);

  std::cout << "size " << scale.values.size() << " " << scale.nMin << " " << scale.nMax << std::endl;
 std::cout << std::endl;


 int n = 5;
 union {
   float numlog;
   int x;
 } tmp;
 tmp.x = 0x3F800000; //set the exponent to 0 so numlog=1.0
 int incr = 1 << (23-n); //amount to increase the mantissa
 int p=std::pow(2.,n);
 std:: cout << p << ": ";
 for(int i=0;i<p;++i)
   {
     std::cout << tmp.numlog << ", " << ::log2(tmp.numlog) << " " ;
     //lookup_table[i] = std::log2(tmp.numlog); //save the log value
     tmp.x += incr;
   }
 std:: cout << std::endl;
 std:: cout << std::endl;


  std::cout << scale.bin(0.04f).second << std::endl;
  std::cout << scale.bin(0.4f).second << std::endl;
  std::cout << scale.bin(1000.f).second << std::endl;
  std::cout << scale.bin(1024.f).second << std::endl;
  std::cout << scale.bin(1024.01f).second  << std::endl;
  std::cout << scale.bin(1200.f).second  << std::endl;
  std::cout << scale.bin(1536.f).second  << std::endl;
  std::cout << scale.bin(1900.f).second  << std::endl;
  std::cout << scale.bin(2047.99f).second  << std::endl;

 std::cout << std::endl;

  std::cout << scale.bin(0.04f).first << std::endl;
  std::cout << scale.bin(0.4f).first << std::endl;
  std::cout << scale.bin(1000.f).first << std::endl;
  std::cout << scale.bin(1024.f).first << std::endl;
  std::cout << scale.bin(1024.01f).first  << std::endl;
  std::cout << scale.bin(1200.f).first  << std::endl;
  std::cout << scale.bin(1536.f).first  << std::endl;
  std::cout << scale.bin(1900.f).first  << std::endl;
  std::cout << scale.bin(2047.99f).first  << std::endl;
  std::cout << scale.bin(10000.f).first << std::endl;


  std::cout << std::endl;

  std::cout << scale.value(0.1f) << std::endl;
  std::cout << scale.value(1.f) << std::endl;
  std::cout << scale.value(100.f) << std::endl;
  std::cout << scale.value(1000.f) << std::endl;
  std::cout << scale.value(1024.f) << std::endl;
  std::cout << scale.value(1024.01f) << std::endl;
  std::cout << scale.value(1200.01f) << std::endl;
  std::cout << scale.value(10000.f) << std::endl;


  return 0;
}
