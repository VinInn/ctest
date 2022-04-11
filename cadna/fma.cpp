#include <cadna.h>
#include <cmath>
#include <cstdio>

int main()
{
  double_st a,b,c,d,e;
  cadna_init(-1);
  a = b = 884279719003555.0;
  /*
  a=10.;
  b=2.;
  c=3.;
  d=4;
  */
  // c = -b*b;
  e=fma(a,a,-b*b);
  printf("e= %s\n",strp(e));
  cadna_end();
 }

