#include<cmath>
void scalar(float& a, float& b) {
  a = std::sqrt(a);
  b = 1.f/b;
}

float half(float a) {
  return a/3.3f;
}

float v[1024];
float w[1024];

void vector() {
  for(int i=0;i!=1024;++i) {
    v[i] = std::sqrt(v[i]);
    w[i] = 1.f/w[i];
  }
}
