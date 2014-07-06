/*
#ifdef __MMX__
#include <mmintrin.h>
#endif

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif
*/

#ifdef __SSE3__
#include <pmmintrin.h>
#endif

#include<iostream>

struct M2 {
  union {
    __m128d r[2];
    double m[4];
  };
  
  double & operator[](int i) { return m[i];}
  __m128d & r0() { return r[0]; }
  __m128d & r1() { return r[1]; }
 
  double  operator[](int i) const { return m[i];}
  __m128d const & r0() const { return r[0]; }
  __m128d const & r1() const { return r[1]; }
  
  
  bool invert() {
    //  load 2-3 as 3-2
    __m128d tmp = _mm_shuffle_pd(r1(),r1(),1);
    // mult and sub
    __m128d det = _mm_mul_pd(r0(),tmp);
    __m128d det2 = _mm_shuffle_pd(det,det,1);
    // det  and -det 
    det = _mm_sub_pd(det,det2);
    // m0 /det, m1/-det -> m3, m2
    r1() = _mm_div_pd(r0(),det);
    r1() = _mm_shuffle_pd(r1(),r1(),1);
    // m3/det, m2/-det -> m0 m1
    r0()=  _mm_div_pd(tmp,det);
    double d; _mm_store_sd(&d,det);
    return !(0.==d);
  } 
  
}  __attribute__ ((aligned (16))) ;


void print(M2 const & m) {
  for (int i=0; i<4; i++) std::cout << m[i] << ", ";
  std::cout << std::endl;

}

void print(bool nofail, M2 const & m) {
  if (!nofail) std::cout << "inversion failed" << std::endl;
  ::print(m);
}

bool invertSym(double m[3]) {

  double det = m[0]*m[2]-m[1]*m[1];
  m[1] /= -det; 
  double tmp = m[0];
  m[0] = m[2]/det;
  m[2] = tmp/det;
  return !(det==0);
}

bool invert(double m[4]) {

  double det = m[0]*m[3]-m[1]*m[2];
  double tmp =m[1];
  m[1] = -m[2]/det; 
  m[2] = -tmp/det; 
  tmp = m[0];
  m[0] = m[3]/det;
  m[3] = tmp/det;
  return !(det==0);
}


int main() {

#ifdef  __SSE3__
  std::cout << "sse3 enabled" << std::endl;
#endif

#ifdef __x86_64__
  std::cout << "x86_64 arch" << std::endl;
#endif

  _mm_setcsr (_mm_getcsr () | 0x8040);    // on Intel, treat denormals as zero for full speed
  
  {
    bool ok=true;
    M2 m = { 1.,2.,-2.,0.5};
    /*
    m[0]=1.;
    m[1]=2.;
    m[2]=-2.;
    m[3]=0.5;
    */
    print(m);
    ok = invert(m.m);
    print(ok,m);
    ok = invert(m.m);
    print(ok,m);
    
    ok = m.invert();
    print(ok,m);
    ok = m.invert();
    print(ok,m);

  }
  
  {
    double mm[4] = { 1.,2.,-2.,0.5};
    bool ok;
    M2 & m = *(M2*)(mm);
    print(m);
    ok = invert(mm);
    print(ok,m);
    ok = invert(mm);
    print(ok,m);
    
    ok = m.invert();
    print(ok,m);
    ok = m.invert();
    print(ok,m);
    
  }

 {

    double mm[4] = { 1.,2.,2.,4.};
    bool ok;
    M2 & m = *(M2*)(mm);
    print(m);
    ok = invert(mm);
    print(ok,m);
    ok = invert(mm);
    print(ok,m);
    
    double mm2[4] = { 1.,2.,2.,4.};
    M2 & m2 = *(M2*)(mm2);
    ok = m2.invert();
    print(ok,m2);
    ok = m2.invert();
    print(ok,m2);
    
  }

  return 0;

}
