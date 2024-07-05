#include<cstdio>
#include<cmath>

int main(int h, char**) {
 double x = 0x1.dffffb3488a4p-1;
 x*=h;
 double y = acos (x);
 printf ("x=%la y=%la\n", x, y);
 return 0;
}

