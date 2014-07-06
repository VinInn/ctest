union hexdouble {
  double d;
  struct {
    unsigned long long mant:52;
    unsigned int exp:11;
    unsigned int sign:1;
  } s;
};
inline
void d2hex(double x, unsigned long long & mant, int & e) {
  hexdouble h; h.d=x;
  mant = h.s.mant;
  e = h.s.exp-1023;
}

inline
void hex2d(double & x, unsigned long long mant, int e) {
  hexdouble h;
  h.s.mant=mant;
  h.s.exp =e+1023;
  x=h.d;
}



inline
unsigned long long d2ll(double x) {
  hexdouble h; h.d=x; return h.s.mant;
}

inline
int d2e(double x) {
  hexdouble h; h.d=x; return h.s.exp-1023;
}


double a[1024],b[1024];
float c[1024];
long long ll[1024];
int e[1024];

unsigned long long l1[1024], l2[1024]; int i3[1024];

void foo() {
  for (int i=0;i!=1024;++i)
    l1[i]=d2ll(a[i]);
}
void foo2() {
  for (int i=0;i!=1024;++i)
    e[i]=d2e(a[i]);
}

void bar() {
  for (int i=0;i!=1024;++i)
    d2hex(a[i],l1[i],e[i]);
}

void bar2() {
  for (int i=0;i!=1024;++i)
    hex2d(a[i],l1[i],-1);
}
