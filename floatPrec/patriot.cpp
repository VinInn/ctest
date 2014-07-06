#include<cstdio>


int main() {
   float tenth=0.1f;
   float count = float(60*60*100*10);
   printf("%f %f %a\n",count,count*tenth,count*tenth);	
   float t=0;
   long long n=0;
   while(n<1000000) {
     t+=0.1f;
     ++n;
     if (n<21 || n%36000==0) printf("%d %f %a\n",n,t,t);
   }
   return 0;
}
