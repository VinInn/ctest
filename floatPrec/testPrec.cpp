#include<cstdio>

using Float = double;

int main() {

  const Float t = 1.;
  Float r = 2;
  Float a = 1.;
  const Float h = 0.5;
  while ( t!=r)  {
    r = t + a;
    a*=h;
  } 
  printf("%f %a\n",a,a);

  return 0;
}
