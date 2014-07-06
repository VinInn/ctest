
namespace vec {
  union binary32 {
    binary32() : ui32(0) {};
    binary32(float ff) : f(ff) {};
    binary32(int  ii) : i32(ii){}
    binary32(unsigned int ui) : ui32(ui){}
    
    unsigned int ui32; /* unsigned int */                
    int i32; /* Signed int */                
    float f;
  };

  inline void
  frexp(float x, int & er, float & mr) {
    binary32 xx,m;
    xx.f = x;
    
    // as many integer computations as possible, most are 1-cycle only, and lots of ILP.
    int e= (((xx.i32) >> 23) & 0xFF) -127; // extract exponent
    m.i32 = (xx.i32 & 0x007FFFFF) | 0x3F800000; // extract mantissa as an FP number
   
    /*
    int adjust = (xx.i32>>22)&1; // first bit of the mantissa, tells us if 1.m > 1.5
    m.i32 -= adjust << 23; // if so, divide 1.m by 2 (exact operation, no rounding)
    e += adjust;           // and update exponent so we still have x=2^E*y
    */

    er = e;
    // now back to floating-point
    mr = m.f; 
    // all the computations so far were free of rounding errors...
  }

}

float x[1024];
int   e[1024];
float m[1024];

void foo() {
  for (int i=0;i!=1204;++i)
    vec::frexp(x[i],e[i],m[i]);
}


int sumi() {
  int s=0;  float p=1.f;
  for (int i=0;i!=1204;++i){
    int er=0; float mr=0;
    vec::frexp(x[i],er,mr);
    s+=er; p*=mr;
  }
  int er=0;
  vec::frexp(p,er,p);
  return s + er;
}


int suma() {
  int s=0; int sr=0; float p=1.f;
  for (int i=0;i!=1204;++i){
    int er=0; float mr=0;
    vec::frexp(x[i],er,mr);
    s+=er; p*=mr;
    vec::frexp(p,er,mr);
    sr+=er; //p=mr;
  }
  return s+sr+p;
}
