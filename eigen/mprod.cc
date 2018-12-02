#include <cmath>
#include <Eigen/Core>
#include <tuple>

using v5 = Eigen::Matrix<float,5,1>;
using m5 = Eigen::Matrix<float,5,5>;
using M = Eigen::Map<m5,0,Eigen::Stride<5*1024,1024> >;
using V = Eigen::Map<v5,0,Eigen::InnerStride<1024> >;

/*
m5 foo(m5 const & c, m5 const & j) {
  return j*c*j.transpose();
}
*/

void dot(float * __restrict__ b, float * __restrict__ c, float * __restrict__ r) 
{ 
  #pragma GCC ivdep  
  for (int i=0; i<256;++i) {
    V v1(b+i,5);
    V v2(c+i,5);
    r[i] = v1.dot(v2);
  }
}

void sum(float * __restrict__ b, float * __restrict__ c, float * __restrict__ r)
{
  #pragma GCC ivdep
  for (int i=0; i<256;++i) {
    V v1(b+i,5);
    V v2(c+i,5);
    V v3(r+i,5);
    v3 = v1+v2;
  }
}

void mdot(float * __restrict__ b, float * __restrict__ c, float * __restrict__ r)
{
  constexpr int os = 5*1024;
  #pragma GCC ivdep
  for (int i=0; i<256;++i) {
    V v2(c+i,5);
    V m1(b+i,5);
    r[i] = m1.dot(v2);
    V m2(b+i+os,5);
    r[i+1024] = m2.dot(v2);
    V m3(b+i+2*os,5);
    r[i+2*1024] = m3.dot(v2);
    V m4(b+i+3*os,5);
    r[i+3*1024] = v2.dot(m4);
    V m5(b+i+4*os,5);
    r[i+4*1024] = v2.dot(m5); 
  }
}


void boo(m5 const & c,
    float * __restrict__ j,
    float * __restrict__ r) {
  #pragma GCC ivdep
  for (int i=0; i<256;++i) {
    V jm(j+i,5);
    V rm(r+i,5);
    rm.noalias() = c*jm;
  }
}


void voo(float * __restrict__ c,
    float * __restrict__ j,
    float * __restrict__ r) {
  #pragma GCC ivdep
  for (int i=0; i<256;++i) {
    M cm(c+i,5,5);
    V jm(j+i,5);
    V rm(r+i,5);

    rm.noalias() = cm*jm; 

  }
}

void moo(float * __restrict__ c,
    float * __restrict__ j,
    float * __restrict__ r) {
  #pragma GCC ivdep
  for (int i=0; i<256;++i) {
    M cm(c+i,5,5);
    M jm(j+i,5,5);
    M rm(r+i,5,5);

    rm.noalias() = cm*jm;

  }
}

void bar(float * __restrict__ c, 
    float * __restrict__ j, 
    float * __restrict__ r, float * __restrict__ w) {
  #pragma GCC ivdep
  for (int i=0; i<256;++i) {
    M cm(c+i,5,5);
    M jm(j+i,5,5);
    M rm(r+i,5,5);
    
    M tmp(w+i,5,5);
    // m5 tmp;
    tmp.noalias() = cm*jm.transpose();
    rm.noalias() = jm*tmp;  // cm*jm.transpose();

    }
}

void bar2(float * __restrict__ c,
    float * __restrict__ j,
    float * __restrict__ r, float * __restrict__ w) {
  #pragma GCC ivdep
  for (int i=0; i<256;++i) {
    M cm(c+i,5,5);
    M jm(j+i,5,5);
    M tmp(w+i,5,5);
    // m5 tmp;
    tmp.noalias() = cm*jm.transpose();
//    rm.noalias() = jm*tmp;  // cm*jm.transpose();

  }

  #pragma GCC ivdep
  for (int i=0; i<256;++i) {
    M jm(j+i,5,5);
    M rm(r+i,5,5);
    M tmp(w+i,5,5);
    rm.noalias() = jm*tmp;

    }

}

