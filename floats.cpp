#include<cstdlib>
#include<cstdio>
#include<cmath>
int main() {

 float yp = 32768.f;
 printf("%f %a\n",yp,yp);
 printf("%f %a\n",yp+1.f/512.f,yp+1.f/512.f);
 printf("%f %a\n",yp+1.f/256.f,yp+1.f/256.f);
 printf("%f %a\n",yp+3.f/512.f,yp+3.f/512.f);
 printf("%f %a\n",yp+2.f/256.f,yp+2.f/256.f);
 printf("%f %a\n",yp+0.001f,yp+0.001f);
 printf("%f %a\n",yp+0.01f,yp+0.01f);
 printf("%f %a\n",yp+0.1f,yp+0.1f);
 printf("%f %a\n",yp+1,yp+1);

 float l2 = log10(2.);
 printf("l2 = %f %a\n",l2,l2); 
 float x[] = {1000,1024.01,1025,1500,9000,10000};
 for (int i=0;i!=6;++i) {
   int n;
   float r = frexpf(x[i],&n);
   printf("%f : %f %a %d\n",x[i],r,r,n);
   printf("     %f %a\n",n*l2,n*l2);
 }

 {
   float x = -0.1;
   float z = std::log(1.f+x)*l2;
   float y = x*(1.f+x*(-0.5f+x/3.f))*l2;
   printf("%f %a\n",z,z);
   printf("%f %a\n",y,y);

 }
 return 0;
}
