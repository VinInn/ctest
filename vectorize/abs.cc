#include<cmath>
inline double Abs(double d)
  { return (d >= 0) ? d : -d; }
inline float Abs(float d)
  { return (d >= 0) ? d : -d; }


double sba(double d) {
	return Abs(d);
}

double ssba(double d) {
     return std::abs(d);
}


double a[1024];
double b[1024];

void foo() {
  for (int i=0; i!=1024;++i)
    b[i]=std::abs(a[i]);
}

void afoo() {
  for (int i=0; i!=1024;++i)
    a[i]=std::abs(a[i]);
}


void bar() {
  for (int i=0; i!=1024;++i)
    b[i]=Abs(a[i]);
}

float  c[1024];
float  d[1024];

void foof() {
  for (int i=0; i!=1024;++i)
    c[i]=std::abs(d[i]);
}

void barf() {
  for (int i=0; i!=1024;++i)
    c[i]=Abs(d[i]);
}


