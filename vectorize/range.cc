#include<cmath>
float a,b,c,d;

float z[1024]; bool ok[1024];
constexpr float rBig = 150.;

float kmin[1024],kmax[1024];

void foo() {
  for (int i=0;i!=1024;++i) {
    float rR = a*z[i];
    float rL = b*z[i];
    float rMin = (rR<rL) ? rR : rL;  
    float rMax = (rR<rL) ? rL : rR;  
    float aMin = (rMin>0) ? rMin : rMax;
    float aMax = (rMin>0) ? rMax : rBig;
          aMin = (rMax>0) ? aMin : rBig;
	  kmin[i] = aMin;
	  kmax[i]= aMax;
	  //    ok[i] = aMin-c<aMax+d;  // this is also ok
  }

}



