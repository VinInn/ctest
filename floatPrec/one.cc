#include<cmath>
int one1(float a, float x) {
  return x*a/x;
}

#include<cmath>
int one2(float a, float x) {
  return a*(x/x);
}



int sign1(float a, float x) {
  return x*a/std::abs(x);
}

int sign2(float a, float x) {
  return a*(x/std::abs(x));
}

