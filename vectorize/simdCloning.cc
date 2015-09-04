#pragma omp declare simd notinbranch
float fma(float x,float y, float z) {
   return x+y*z;
}

/*
#define N 100
// #pragma omp declare simd notinbranch
#pragma omp declare simd simdlen(4) notinbranch
long long foo(double c1, double c2)
{
  double z1 = c1, z2 = c2;
  long long res = N;
  #define min(x,y) (x) < (y)? (x): (y)

  for (long long i=0LL; i<N; i++) {
    res = (z1 * z1 + z2 * z2 > 4.0)? min (i,res): res;
    z1 = c1 + z1 * z1 - z2 * z2;
    z2 = c2 + 2.0 * z1 * z2;
  }
  return res;
}

#pragma omp declare simd simdlen(8) notinbranch
int bar(float c1, float c2)
{
  double z1 = c1, z2 = c2;
  int res = N;
  #define min(x,y) (x) < (y)? (x): (y)

  for (int i=0; i<N; i++) {
    res = (z1 * z1 + z2 * z2 > 4.0)? min (i,res): res;
    z1 = c1 + z1 * z1 - z2 * z2;
    z2 = c2 + 2.0 * z1 * z2;
  }
  return res;
}

*/ 

/*
#pragma omp declare simd simdlen(8) notinbranch
float mom(float c1, float c2) {
  #define min(x,y) (x) < (y)? (x): (y)
  return min(c1,c2); 
}
*/
