#include <cmath>

struct XY {
  float x,y;
  float z; int k;
  double w1, w2;
  // double t1, t2;
};


float __attribute__ ((aligned(16))) a[1024];
float __attribute__ ((aligned(16))) b[1024];
float __attribute__ ((aligned(16))) c[1024];
XY __attribute__ ((aligned(16))) x[1024];



void txy() {
  for (int i=0; i!=1024; ++i)
    c[i] = b[i]/a[i];
}

void txys() {
  for (int i=0; i!=1024; ++i)
    c[i] = x[i].y/x[i].x;
}



// does not vectorize
void tVi() {
  for (int i=0; i!=1024; ++i) {
    int j = a[i];
    b[i] = (j==0) ?  a[i]+1 : a[i];
  }
}

// nicely vectorize...
void tVf() {
  for (int i=0; i!=1024; ++i) {
    int j = a[i];
    float z = j;
    b[i] = (z==0) ?  a[i]+1 : a[i];
  }
}

void tVe() {
  for (int i=0; i!=1024; ++i) {
    float z = a[i];
    int n = z;
    n = (n+0x7f)<<23;
 // x = z * n;
    float x = z *   *reinterpret_cast<float*>(&n);
    b[i] = x;
  }
}


void tVe3() {
  for (int i=0; i!=1024; ++i) {
     float z = a[i];
     int n = z;
     c[i] = scalbnf(b[i],n);
  }
}

inline float i2f(int x) {
  union { float f; int i; } tmp;
  tmp.i=x;
 return tmp.f;
}
void tVe2() {
  for (int i=0; i!=1024; ++i) {
    float z = a[i];
    int n = z;
    n = (n+0x7f)<<23;
 // x = z * n;
    float x = z *  i2f(n);
    b[i] = x;
  }
}


void tVP() {
  for (int i=0; i!=1024; ++i) {
    float z = a[i];
    float x = 0.f;
    if (z>1.f) 
      x =(0.1f * z 
	  + 1.3f) * z
	+ z;
    else
      x =(0.2f * z 
	  + 1.3f) * z
	+ z;   
    b[i] = x;
  }
  
}

void tVP2() {
  for (int i=0; i!=1024; ++i) {
    float z = a[i];
    float x = 0.f;
    if (z>1.f) 
      x =(0.1f * z 
	  + 1.3f) * z;
    else
      x =(0.2f * z 
	  + 1.3f) * z;
    b[i] = x+z;
  }
  
}


static constexpr float lut[4] = { 1.f,2.f,3.f,4.f};

void lu() {
  for (int i=0; i!=1024; ++i) {
    int z = a[i];
    b[i] = lut[z];
  }
}
