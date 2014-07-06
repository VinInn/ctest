#include "cephes.h"




constexpr int NVAL = 4096;
float __attribute__ ((aligned(16))) a0[NVAL];
float __attribute__ ((aligned(16))) s0[NVAL];
float __attribute__ ((aligned(16))) c0[NVAL];

float __attribute__ ((aligned(16))) a[NVAL];
float __attribute__ ((aligned(16))) s[NVAL];
float __attribute__ ((aligned(16))) c[NVAL];


float __attribute__ ((aligned(16))) b0[NVAL];
float __attribute__ ((aligned(16))) e0[NVAL];
float __attribute__ ((aligned(16))) l0[NVAL];

float __attribute__ ((aligned(16))) e[NVAL];
float __attribute__ ((aligned(16))) l[NVAL];


void scS() {
  for (int i=0; i!=NVAL; ++i) {
    c0[i] = ::cosf(a0[i]);  
    s0[i] = ::sinf(a0[i]);  
  }
}

void aS() {
  for (int i=0; i!=NVAL; ++i) {
    float z = i+1;
    a0[i] = ::atan2f(z*s0[i],z*c0[i]);  
  }
}

void scV() {
  for (int i=0; i!=NVAL; ++i)
    cephes::sincosf(a0[i], s[i], c[i]);
}

void aV() {
  for (int i=0; i!=NVAL; ++i) {
    float z = i+1;
    a[i] = cephes::atan2f(z*s0[i],z*c0[i]);  
  }
}


void eS() {
  for (int i=0; i!=NVAL; ++i)
    e0[i] = ::expf(b0[i]);
}


void lS() {
  for (int i=0; i!=NVAL; ++i)
    l0[i] = ::logf(e0[i]);
  
}


void hS() {
  for (int i=0; i!=NVAL; ++i)
    e0[i] = ::sinhf(b0[i]);
}


void ahS() {
  for (int i=0; i!=NVAL; ++i)
    l0[i] = ::asinhf(e0[i]);
  
}


void eV() {
  for (int i=0; i!=NVAL; ++i)
    e[i] = cephes::expf(b0[i]);
  
}


void lV() {
  for (int i=0; i!=NVAL; ++i)
    l[i] = cephes::logf(e0[i]);
  
}
  
void hV() {
  for (int i=0; i!=NVAL; ++i)
    e[i] = cephes::sinhf(b0[i]);
  
}


void ahV() {
  for (int i=0; i!=NVAL; ++i)
    l[i] = cephes::asinhf(e0[i]);
  
}
  


#include<cstdio>

inline volatile unsigned long long int rdtsc() {
  volatile unsigned long long int x;
  __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
  return x;
}


void dump(float x) {
  printf("%e:   %e %e %e %e %e %e %e\n",
	 x,::cosf(x),::sinf(x),::atanf(x),::expf(x),::logf(x),::sinhf(x),::asinhf(x)
	 );
  using namespace cephes;
  float s,c; cephes::sincosf(x,s,c);
  printf("%e:   %e %e %e %e %e %e %e\n\n",
	 x,c,s,cephes::atanf(x),cephes::expf(x),cephes::logf(x),cephes::sinhf(x),cephes::asinhf(x)
	 );
}



void timing() {
  volatile double assS=0, tscS=0,tscV=0,taS=0,taV=0,
    teS=0,teV=0,tlS=0,tlV=0,
    thS=0,thV=0,tahS=0,tahV=0;

  for (int k=0;k!=100;++k)  {
    using namespace cephes::cephes_details;
    
    auto volatile t1 =  rdtsc();
    
    a0[0] = -40.f*PIF;
    for (int i=1; i!=NVAL; ++i)
      a0[i] = a0[i-1] + 0.02*PIF;
    assS +=  (rdtsc() -t1);
  

    t1 =  rdtsc();
    scS();
    tscS += (rdtsc() -t1);
  
    t1 =  rdtsc();
    scV();
    tscV +=  (rdtsc() -t1);

    t1 =  rdtsc();
    aS();
    taS +=  (rdtsc() -t1);
  
    t1 =  rdtsc();
    aV();
    taV +=  (rdtsc() -t1);
  
    b0[0] = -102.4;
    for (int i=1; i!=NVAL; ++i)
      b0[i] = b0[i-1] + 0.05;
    
    
    t1 =  rdtsc();
    eS();
    teS +=  (rdtsc() -t1);
    
    t1 =  rdtsc();
    eV();
    teV +=  (rdtsc() -t1);
    
    
    
    t1 =  rdtsc();
    lS();
    tlS +=  (rdtsc() -t1);
    
    t1 =  rdtsc();
    lV();
    tlV +=  (rdtsc() -t1);
    

    t1 =  rdtsc();
    hS();
    thS +=  (rdtsc() -t1);
    
    t1 =  rdtsc();
    hV();
    thV +=  (rdtsc() -t1);
    
    
    t1 =  rdtsc();
    ahS();
    tahS +=  (rdtsc() -t1);
    
    t1 =  rdtsc();
    ahV();
    tahV +=  (rdtsc() -t1);

  }
  
  printf("sincos   atan2  exp   log   sinh   asinh\n");
  printf("%g %g   %g %g   %g %g   %g %g   %g %g   %g %g\n",
	 tscS,tscV,taS,taV,
	 teS,teV,tlS,tlV,
	 thS,thV,tahS,tahV);
  printf("%g   %g   %g   %g   %g   %g\n",
	 double(tscS)/double(tscV),double(taS)/double(taV),
	 double(teS)/double(teV),double(tlS)/double(tlV),
	 double(thS)/double(thV),double(tahS)/double(tahV)
	 );
  
}


int main() {
  using namespace cephes::cephes_details;
  
 auto volatile t1 =  rdtsc();
 
  a0[0] = -40.f*PIF;
  for (int i=1; i!=NVAL; ++i)
    a0[i] = a0[i-1] + 0.02*PIF;
  auto assS =  rdtsc() -t1;
  

  t1 =  rdtsc();
  scS();
  auto tscS =  rdtsc() -t1;
  
  t1 =  rdtsc();
  scV();
  auto tscV =  rdtsc() -t1;
  
  printf("\nsincos\n");
  for (int i=0; i!=NVAL; ++i)
    if (fabs(s0[i]-s[i])>2e-07 || fabs(c0[i]-c[i])>2e-07)
      // if (fabs(b[i])!=fabs(c[i]))
      printf("%d %e %e %e %e %e\n", i,a0[i],s0[i],s[i], c0[i],c[i]);
  

  t1 =  rdtsc();
  aS();
  auto taS =  rdtsc() -t1;
  
  t1 =  rdtsc();
  aV();
  auto taV =  rdtsc() -t1;
  
  printf("\natan\n");
  for (int i=0; i!=NVAL; ++i)
    if (fabs(a[i]-a0[i])>2.4e-07)
      printf("%d %e %e %e\n", i, a0[i],a[i],a0[i]-a[i]);

  printf("%u   %u %u   %u %u\n",assS,tscS,tscV,taS,taV);
  

  //----------------------------------------------------------

  b0[0] = -102.4;
  for (int i=1; i!=NVAL; ++i)
    b0[i] = b0[i-1] + 0.05;


  t1 =  rdtsc();
  eS();
  auto teS =  rdtsc() -t1;
  
  t1 =  rdtsc();
  eV();
  auto teV =  rdtsc() -t1;



  t1 =  rdtsc();
  lS();
  auto tlS =  rdtsc() -t1;
  
  t1 =  rdtsc();
  lV();
  auto tlV =  rdtsc() -t1;

  using cephes::f2i;
  printf("\nexp\n");
  for (int i=0; i!=NVAL; ++i)
    if (std::abs(f2i(e[i])-f2i(e0[i]))>50)
      printf("%e %e %e %e %d\n", b0[i], e0[i],e[i],e0[i]-e[i],f2i(e[i])-f2i(e0[i]));

  int nok=0;
  printf("\nlog\n");
  for (int i=0; i!=NVAL; ++i)
    if ( e0[i]>0 && e0[i]< std::numeric_limits<float>::infinity()) {
      nok++;
      if (std::abs(f2i(l[i])-f2i(l0[i]))>50)
	printf("%e %e %e %e %d\n", e0[i], l0[i],l[i],l0[i]-l[i],f2i(l[i])-f2i(l0[i]));
    }
  printf("\nlog %d\n",nok);

  //-----------------------------------------------------------------

  t1 =  rdtsc();
  hS();
  auto thS =  rdtsc() -t1;
  
  t1 =  rdtsc();
  hV();
  auto thV =  rdtsc() -t1;


  t1 =  rdtsc();
  ahS();
  auto tahS =  rdtsc() -t1;
  
  t1 =  rdtsc();
  ahV();
  auto tahV =  rdtsc() -t1;

  using cephes::f2i;
  printf("\nsinh\n");
  nok=0;
  for (int i=0; i!=NVAL; ++i)
    if (std::abs(f2i(e[i])-f2i(e0[i]))>50)
      printf("%e %e %e %e %d\n", b0[i], e0[i],e[i],e0[i]-e[i],f2i(e[i])-f2i(e0[i]));

  printf("\nasinh\n");
  for (int i=0; i!=NVAL; ++i)
    if ( e0[i]>0 && e0[i]< 1.9e+19) { // std::numeric_limits<float>::infinity()) {
      nok++;
      if (std::abs(f2i(l[i])-f2i(l0[i]))>50)
	printf("%e %e %e %e %d\n", e0[i], l0[i],l[i],l0[i]-l[i],f2i(l[i])-f2i(l0[i]));
    }
  printf("\nasinh %d\n",nok);


  printf("sincos   atan2  exp   log   sinh   asinh\n");
  printf("%u %u   %u %u   %u %u   %u %u   %u %u   %u %u\n",
	 tscS,tscV,taS,taV,
	 teS,teV,tlS,tlV,
	 thS,thV,tahS,tahV);
  printf("%g   %g   %g   %g   %g   %g\n",
	 double(tscS)/double(tscV),double(taS)/double(taV),
	 double(teS)/double(teV),double(tlS)/double(tlV),
	 double(thS)/double(thV),double(tahS)/double(tahV)
	 );

  
  printf("\ntest extreme values\n");
  constexpr float pif = 3.141592653589793238f;
  const float exval[] = {
    0.f, 1.f, -1.f
    ,0.5*pif,  -0.5*pif, pif, -pif, 1.5*pif,  -1.5*pif, 2.f*pif, -2.f*pif 
    ,std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()
    ,std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()
  };
  for (int i=0; i!=sizeof(exval)/sizeof(float); ++i)
    dump(exval[i]);



  // double for atan2
  for (int i=0; i!=sizeof(exval)/sizeof(float); ++i) {
    printf("%e  : ",  exval[i]);
    for (int j=0; j!=sizeof(exval)/sizeof(float); ++j)
      printf(" %e: %e %e,", exval[j], ::atan2f(exval[i],exval[j]), cephes::atan2f(exval[i],exval[j]));
    printf("\n");
  }


  timing();
  return 0;
  
}
