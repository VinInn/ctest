#include <iostream>
#include <cmath>
#include <vector>
#include <fwBase.h>
#include <fwSignal.h>
#include <ippvm.h>

void compute_f(float const * __restrict__ v, float * __restrict__ r, int buckets)
{
 for(int i=0;i!=buckets; ++i) r[i]=std::log10(v[i]);
}

void computeFWA11_f(float const * __restrict__ v, float * __restrict__ r, int buckets)
{
 fwsLog10_32f_A11(v,r,buckets);
}

void computeIPPA11_f(float const * __restrict__ v, float * __restrict__ r, int buckets)
{
 ippsLog10_32f_A11(v,r,buckets);
}

void compute_d(double const * __restrict__ v, double * __restrict__ r, int buckets)
{
 for(int i=0;i!=buckets; ++i) r[i]=std::log10(v[i]);
}

void computeFWA11_d(double const * __restrict__ v, double * __restrict__ r, int buckets)
{
 fwsLog10_64f_A50(v,r,buckets);
}

void computeIPPA11_d(double const * __restrict__ v, double * __restrict__ r, int buckets)
{
 ippsLog10_64f_A50(v,r,buckets);
}


void triggerup(){}
void triggerdown(){}

void fill_vector_f(std::vector<float> &v, int buckets, double end)
{
 float fix = log10f((float)end)/(buckets-1);
 for(int i=0;i!=buckets; ++i) v.at(i) = powf(10, fix*i);
 return;
}

void fill_vector_d(std::vector<double> &v, int buckets, double end)
{
 double fix = log10(end)/(buckets-1);
 for(int i=0;i!=buckets; ++i) v.at(i) = pow(10, fix*i);
 return;
}

int main(int argc, char * argv[])
{
 if(argc!=5)
 {
  printf("\nUSAGE: %s [f,d] [s,f,i] #no_buckets, upper_limit\nExample: %s f s 7 1000\n\n", argv[0], argv[0]);
  exit(1);
 }

 int buckets = atoi(argv[3]);

 std::vector<float> v_f(buckets, 0.);
 std::vector<float> r_f(buckets, 0.);
 std::vector<double> v_d(buckets, 0.);
 std::vector<double> r_d(buckets, 0.);

 double end = atof(argv[4]);

 fill_vector_f(v_f, buckets, end);
 fill_vector_d(v_d, buckets, end);

 //for(int i=0;i!=buckets; ++i) printf("%d: %f\n", i, v_f.at(i));

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
   compute_f(&v_f.front(),&r_f.front(), buckets);
  }
  triggerdown();
  for(int i=0;i!=buckets; ++i) std::cout << r_f[i] << ", ";
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
   computeFWA11_f(&v_f.front(),&r_f.front(), buckets);
  }
  triggerdown();
  for(int i=0;i!=buckets; ++i) std::cout << r_f[i] << ", ";
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
   computeIPPA11_f(&v_f.front(),&r_f.front(), buckets);
  }
  triggerdown();
  for(int i=0;i!=buckets; ++i) std::cout << r_f[i] << ", ";
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
   compute_d(&v_d.front(),&r_d.front(), buckets);
  }
  triggerdown();
  for(int i=0;i!=buckets; ++i) std::cout << r_d[i] << ", ";
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
   computeFWA11_d(&v_d.front(),&r_d.front(), buckets);
  }
  triggerdown();
  for(int i=0;i!=buckets; ++i) std::cout << r_d[i] << ", ";
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
   computeIPPA11_d(&v_d.front(),&r_d.front(), buckets);
  }
  triggerdown();
  for(int i=0;i!=buckets; ++i) std::cout << r_d[i] << ", ";
  std::cout << std::endl;
 }
}
