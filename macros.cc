#define A2(a,b) a+b
#define A3(a,b,c) a+b+c


int sum(int x, int y) { return A2(x,y);}
int sum(int x, int y, int z) { return A3(x,y,z);}

