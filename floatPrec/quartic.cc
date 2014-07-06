#include <cmath>

float a[1024],b[1024],c[1024], x1[1024],x2[1024];



void solver() {

   for (int i=0; i!=1024; ++i) {
    double det = double(b[i])*double(b[i])-4*double(a[i])*double(c[i]);
    float q = -0.5f*(std::copysign(std::sqrt(float(det)),b[i])+b[i]);
    x1[i] = q/a[i];
    x2[i] = c[i]/q;
   }

}
