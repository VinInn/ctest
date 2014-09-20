#include <cmath>
#include<cstdio>

int main(int n, char * v[]) {
   float fn = n - 1.f;
   float k = 0.1f + fn;

   float q = 1.f/std::sqrt(k);
   printf("%a %a\n",k,q);
   return 0;
}
