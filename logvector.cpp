#include <iostream>
#include <cmath>
#include <vector>
#include <fwBase.h>
#include <fwSignal.h>
#include <ippvm.h>

std::vector<float> v_f(4,0.);
std::vector<float> r_f(4,0.);
std::vector<double> v_d(4,0.);
std::vector<double> r_d(4,0.);


void compute_f(float const * __restrict__ v, float * __restrict__ r)
{
 for(int i=0;i!=4; ++i) r[i]=std::log10(v[i]);
}

void computeFWA11_f(float const * __restrict__ v, float * __restrict__ r)
{
 fwsLog10_32f_A11(v,r,4);
}

void computeIPPA11_f(float const * __restrict__ v, float * __restrict__ r)
{
 ippsLog10_32f_A11(v,r,4);
}

void compute_d(double const * __restrict__ v, double * __restrict__ r)
{
 for(int i=0;i!=4; ++i) r[i]=std::log10(v[i]);
}

void computeFWA11_d(double const * __restrict__ v, double * __restrict__ r)
{
 fwsLog10_64f_A50(v,r,4);
}

void computeIPPA11_d(double const * __restrict__ v, double * __restrict__ r)
{
 ippsLog10_64f_A50(v,r,4);
}


void triggerup(){}
void triggerdown(){}


int main(int argc, char * argv[])
{
 v_f.at(0) = 1.;
 v_f.at(1) = 10.;
 v_f.at(2) = 100.;
 v_f.at(3) = 1000.;

 v_d.at(0) = 1.;
 v_d.at(1) = 10.;
 v_d.at(2) = 100.;
 v_d.at(3) = 1000.;

 int const nloops=10000000;

 if(argv[1][0]=='f' && argv[2][0]=='s')
 {
  triggerup();
  for(int i=0; i!=nloops; ++i)
  {
   __asm__ __volatile__ ("# Trick the compiler."
                         : "=o"(v_f),"=o"(r_f)
                         : "o"(v_f),"o"(r_f)
                         : );
   compute_f(&v_f.front(),&r_f.front());
  }
  triggerdown();
  for(int i=0;i!=4; ++i) std::cout << r_f[i] << ", ";
  std::cout << std::endl;
 }
 else if(argv[1][0]=='f' && argv[2][0]=='f')
 {
  triggerup();   
  for(int i=0; i!=nloops; ++i)
  {
   __asm__ __volatile__ ("# Trick the compiler."
                         : "=o"(v_f),"=o"(r_f)
                         : "o"(v_f),"o"(r_f)
                         : );
   computeFWA11_f(&v_f.front(),&r_f.front());
  }
  triggerdown();
  for(int i=0;i!=4; ++i) std::cout << r_f[i] << ", ";
  std::cout << std::endl;
 }
 else if(argv[1][0]=='f' && argv[2][0]=='i')
 {
  triggerup();   
  for(int i=0; i!=nloops; ++i)
  {
   __asm__ __volatile__ ("# Trick the compiler."
                         : "=o"(v_f),"=o"(r_f)
                         : "o"(v_f),"o"(r_f)
                         : );
   computeIPPA11_f(&v_f.front(),&r_f.front());
  }
  triggerdown();
  for(int i=0;i!=4; ++i) std::cout << r_f[i] << ", ";
  std::cout << std::endl;
 }
 else if(argv[1][0]=='d' && argv[2][0]=='s')
 {
  triggerup();
  for(int i=0; i!=nloops; ++i)
  {
   __asm__ __volatile__ ("# Trick the compiler."
                         : "=o"(v_d),"=o"(r_d)
                         : "o"(v_d),"o"(r_d)
                         : );
   compute_d(&v_d.front(),&r_d.front());
  }
  triggerdown();
  for(int i=0;i!=4; ++i) std::cout << r_d[i] << ", ";
  std::cout << std::endl;
 }
 else if(argv[1][0]=='d' && argv[2][0]=='f')
 {
  triggerup();   
  for(int i=0; i!=nloops; ++i)
  {
   __asm__ __volatile__ ("# Trick the compiler."
                         : "=o"(v_d),"=o"(r_d)
                         : "o"(v_d),"o"(r_d)
                         : );
   computeFWA11_d(&v_d.front(),&r_d.front());
  }
  triggerdown();
  for(int i=0;i!=4; ++i) std::cout << r_d[i] << ", ";
  std::cout << std::endl;
 }
 else if(argv[1][0]=='d' && argv[2][0]=='i')
 {
  triggerup();   
  for(int i=0; i!=nloops; ++i)
  {
   __asm__ __volatile__ ("# Trick the compiler."
                         : "=o"(v_d),"=o"(r_d)
                         : "o"(v_d),"o"(r_d)
                         : );
   computeIPPA11_d(&v_d.front(),&r_d.front());
  }
  triggerdown();
  for(int i=0;i!=4; ++i) std::cout << r_d[i] << ", ";
  std::cout << std::endl;
 }
}
