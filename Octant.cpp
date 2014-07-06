#include<cmath>
#include<random>

#include<cstdio>
#include<iostream>
/*
o     01234567
ax>ay 10011001
x>0   11110000
y>0   00111100
x>y   10000111
-x>y  00011110
q1    11332200
q2    10022331
*/

int octantIndex(float x, float y) {
  return (y<0)*4 + (x<0)*2 + (fabs(x)<fabs(y));
}

int octant(int i) {
  constexpr int o[] {0,1,3,2,7,6,4,5};
  return o[i];
}

int qua1(float x, float y) {
  return (x>0) + 2*(y>0);
}

int qua2(float x, float y) {
  return (x>y) + 2*((-x)>y);
}

bool sameQuadrant(float x1, float y1, float x2, float y2) {
  return ( qua1(x1,y1)==qua1(x2,y2) ) | ( qua2(x1,y1)==qua2(x2,y2) );
}

bool sameQuadrant(int lq1, int lq2, int rq1, int rq2) {
  return lq1==rq1 | lq2==rq2;
} 

bool sameQuadrant(int i, int j) {
  int k = abs(i-j);
  return k<2 || k==7;
}

int main() {
  const float PIF = 3.141592653589793238;

  for (float x=-2; x<2.2; x+=0.5)
    for (float y=-2.6; y<2.6; y+=0.5) {
      auto in = octantIndex(x,y);
      float oo = 4.f*std::atan2(y,x)/PIF; if (oo<0) oo = 8+oo;
      printf("%f %f %f %d %d %d\n",x,y, std::atan2(y,x), int(oo), in,octant(in));
    }


   std::mt19937 eng;
   std::uniform_real_distribution<float> rgen(-5.,5.);

   //std::cout << rgen(eng) << std::endl;

  for (int i=0; i!=100000; ++i) {
    float x1=rgen(eng);
    float y1=rgen(eng);
    float x2=rgen(eng);
    float y2=rgen(eng);
    float p1 = std::atan2(y1,x1);
    float p2 = std::atan2(y2,x2);
    float dp = std::abs(p2-p1);
    if (dp>PIF) dp = (2.f*PIF)-dp;
    if (dp<(PIF/4.f)  && !sameQuadrant( x1,y1, x2,y2) ) printf("%f %f %f\n", p1*180./PIF, p2*180./PIF, dp*180./PIF);
    auto o1 = octant(octantIndex(x1,y1));
    auto o2 = octant(octantIndex(x2,y2));
    if (dp<(PIF/4.f)  && !sameQuadrant(o1,o2) ) printf("%f %f %f\n", p1*180./PIF, p2*180./PIF, dp*180./PIF);

  }

  return 0;

}
