// >>>fastFit: 0=0x1.2ce8882e63f3cp+13 1=0x1.6fb16ff9f82p+12 2=0x1.609e8cc4f9e0bp+13 3=0x1.51d2c34a642b4p-2 tmp=-0x1.08adb10b6ea2ep+8 a=(-0x1.1d4a12p+1,0x1.d3c23p+1) b=(-0x1.bdd16p-1,0x1.6db27p+0) c=(0x1.8cbe6ap+1,-0x1.454db4p+2)
// >>>fastFit: 0=-0x1.9e2867d7e4a8fp+14 1=-0x1.32af012ef8667p+16 2=0x1.43b0b33b0e313p+16 3=0x1.79ad18c7c87dbp-1 tmp=0x1.359f981cfc0c5p+7 a=(-0x1.e4e9eap+2,0x1.4755cdp+1) b=(-0x1.35bda8p+2,0x1.a21084p+0) c=(0x1.8d53c9p+3,-0x1.0c2f078p+2)
// >>>fastFit: 0=-0x1.3a5261ca30c68p+15 1=-0x1.0ff03d7c7083ap+17 2=0x1.1b106bc6497cap+17 3=-0x1.08eb16ba5d4b3p-2 tmp=0x1.230d3d5343f9dp+10 a=(-0x1.ad3c79p+2,0x1.f00e2ap+0) b=(-0x1.e8f71p+0,0x1.1a822p-1) c=(0x1.13bd1e8p+3,-0x1.3ea79dp+1)
// >>>fastFit: 0=0x1.156d3dcc0978dp+25 1=0x1.729941c4af08ap+26 2=0x1.8bb53ac0f4cd6p+26 3=-0x1.02930d0dbe695p+1 tmp=-0x1.99bc2fb830c9dp+17 a=(-0x1.fd6b2cp+1,0x1.7d589cp+0) b=(-0x1.09d10dp+3,0x1.8df9dp+1) c=(0x1.892bd8p+3,-0x1.26530fp+2)
// >>>fastFit: 0=inf 1=inf 2=inf 3=-nan tmp=inf a=(0x1.79a72p-2,-0x1.e1588p+1) b=(0x1.de4706p-1,-0x1.30ccacp+3) c=(-0x1.4d8d4bp+0,0x1.a922ccp+3)

#include <iostream>
#include<cmath>

// This function is here purely to break the optimizer by pretending to clobber
// the memory it's pointed at.  It compiles to a single "ret" instruction, and
// when inlined optimizes away into nothing.
void invalidate(void* ptr) {
  asm (
      "" /* no instructions */
      : /* no inputs */
      : /* output */ "rax"(ptr)
      : /* pretend to clobber */ "memory"
      );
}


template<typename T>
void testit(T * a, T * b) {
 invalidate(a);
 invalidate(b);
  auto x1 = a[0]*b[1] - a[1]*b[0];
  auto c1 = a[0]*b[1];
  auto c2 = a[1]*b[0];
  invalidate(&c1);
  auto x2 = c1 - c2;
  auto x3 = std::fma(a[0],b[1], - a[1]*b[0]);
  auto x4 = std::fma(-a[1],b[0], a[0]*b[1]);
  

  std::cout << double(x1) << std::endl;
  std::cout << double(x2) << std::endl;
  std::cout << double(x3) << std::endl;
  std::cout << double(x4) << std::endl;
  std::cout << std::endl;
}



int main(int n, char**) {

//  double a[] = {0.368802,-3.760513};
//  double b[] = {0.934136,-9.524984};
//  double c[] = {-1.302937,13.285498};

//  double a[] ={-0x1.fd6b2cp+1,0x1.7d589cp+0};
//  double b[] ={-0x1.09d10dp+3,0x1.8df9dp+1};
//  double c[] ={0x1.892bd8p+3,-0x1.26530fp+2};

    double a[] = {0x1.79a72p-2,-0x1.e1588p+1};
    double b[] = {0x1.de4706p-1,-0x1.30ccacp+3};
    double c[] = {-0x1.4d8d4bp+0,0x1.a922ccp+3};


  invalidate(a);
  if (n>3) {
    a[1] += 3;
    a[0] += 4.12;
    c[0] = a[1];
    b[1] = c[1];
  }

 testit(a,b);
 testit(a,c);
 testit(b,c);

  return 0;
}

