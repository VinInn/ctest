#include<iostream>
#include<cstdio>
#include<cassert>

  union binary32 {
    binary32() : ui32(0) {};
    binary32(float ff) : f(ff) {};
    binary32(int32_t ii) : i32(ii){}
    binary32(uint32_t ui) : ui32(ui){}
    
    uint32_t ui32; /* unsigned int */                
    int32_t i32; /* Signed int */                
    float f;
  };

int main() {

  unsigned int m = 0xffffffff;
  int sm = 0x7fffffff;

  float p1=0;
  double s1=0; double d1=0;
  float p2=0;
  double s2=0; double d2=0;
  float p3=0;
  double s3=0; double d3=0;
  float m3=1000.f, x3=0;

  binary32 h(0.5f);
  
  printf("%x\n",h.ui32);
  printf("%x\n",h.ui32&0x007FFFFF);
  printf("%x\n",~0x807FFFFF);
  printf("%e\n",double(0x007FFFFF));

  for (unsigned int i=1; i!=m; ++i) {
    binary32 w(i);
    float f1 = float(w.ui32)/(float(m));
    if (f1==p1) s1++; else d1++;
    p1 = f1;

    float f2 = float(w.i32)/(float(sm));
     //   float f2 = 0.5f+float(w.i32)/(2.f*float(sm));
    if (f2==p2) s2++; else d2++;
    p2 = f2;


    // 0  
    int n = 8-__builtin_clz(w.ui32|0x007FFFFF );
    assert(n>=-1 && n<9);
    w.ui32 &= 0x007FFFFF; w.ui32 |=h.ui32;
    if (n>-1) w.ui32 += (n<<23);
    float f3 = (n==-1) ? w.f-0.5f : w.f;
    m3 = std::min(m3,f3);
    x3 = std::max(x3,f3);
    if (f3==p3) s3++; else d3++;
    p3 = f3;

  }
  std ::cout << double(m) << " " << s1 << " " << d1 << std::endl;
  std ::cout << double(m) << " " << s2 << " " << d2 << std::endl;
  std ::cout << double(m) << " " << s3 << " " << d3 << " " << m3 << " " << x3 << std::endl;


}
