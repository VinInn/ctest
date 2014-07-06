#include<cmath>
#include<algorithm>


inline float foo(float a, float b) {

  float x = std::floor(a);
  float y = 1./std::sqrt(x);
  float z = a/b;
  float k = 1./b;
  float w = std::min(a,b);
  return (x+y+z)/std::sqrt(k*k+w*w);

}



float v1[1024];
float v2[1024];
float v3[1024];



void scalar() {
  v1[0]=foo(v2[0],v3[0]);
}

void vect() {
  for (int i=0;i!=1024;++i)
    v1[i]=foo(v2[i],v3[i]);
}
