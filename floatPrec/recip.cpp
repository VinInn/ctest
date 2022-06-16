#include <cmath>
#include<cstdio>

int main(int n, char * v[]) {

   printf("%a %a\n", 0.1f,10.0f);

   float fn = (n-1);
   float k = 0.1f + fn;

   float x[32]; float y[32];
   for ( auto & a : x) a = k;

   float q = std::sqrt(k);
   printf("%a %a\n",k,q);
   k+=fn;
   q = 1.f/std::sqrt(k);
   printf("%a %a\n",k,q);
   k+=fn;
   q = 1.f/k;
   printf("%a %a\n",k,q);

   int i=0;
   for ( auto const &	a : x) y[i++] = std::sqrt(a);
   printf("%a %a\n",k,y[5]);
   i=0;
   for ( auto const &   a : x) y[i++] =	1/std::sqrt(a);
   printf("%a %a\n",k,y[5]);
   i=0;
   for ( auto const &   a : x) y[i++] =	1/a;
   printf("%a %a\n",k,y[5]);

}
