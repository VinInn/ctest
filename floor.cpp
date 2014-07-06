#include<cmath>
#include<iostream>

int main() {

for (float x=-2.45; x<3; ++x)
   std::cout << x << " " << floorf(x) << " " << int(x) << " " 
             << copysignf(float(int(fabs(x))),x) <<std::endl;


for (float x=-2.45; x<-1.45; x+=0.1)
   std::cout << x << " " << floorf(x+0.5f) << " " << int(x+0.5f) << " "
             << copysignf(float(int(fabs(x+0.5f))),x+0.5f) <<std::endl;

}


