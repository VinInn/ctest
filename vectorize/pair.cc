constexpr int N = 1024;
constexpr int H = N/2;

inline void foo(float x, float &c, float &s) {
   c = 0.5f*x*x;
   s=  x;
}

inline void goo(float &x, float &y) {
  float u = y*y;
  float c,s; foo(x,c,s);
  x = u*c; y=u*s;  
}

inline void goo2 (float a, float b, float &x, float &y) {
  float u = a*a;
  float c,s; foo(b,c,s);
  x = u*c; y=u*s;
}


float a[1024];
float b[1024];


void bar() {
  for (int i=0; i!=H; ++i) 
    foo(a[i],b[i],b[H+i]);
}

void bar2() {
  for (int i=0; i!=H; ++i) {
    b[i] = a[i]; b[H+i] = a[H+i];
    goo(b[i],b[H+i]);
  }
}


void bar3() {
  for (int i=0; i!=H; ++i) {
    goo2(a[i],a[H+i], b[i],b[H+i]);
  }
}

