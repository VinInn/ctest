#include<cstdio>

using FLOAT = double;

int main() {
   FLOAT tenth=0.1;
   FLOAT count = FLOAT(60*60*100*10);
   printf("%f %f %a\n",count,count*tenth,count*tenth);	
   FLOAT t=0;
   long long n=0;
   while(n<10000000) {
     t+=0.1f;
     ++n;
     if (n<21 || n%36000==0) printf("%lld %f %a\n",n,t,t);
   }
   return t;
}
