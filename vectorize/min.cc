// #include<utililies>
#include<memory>
float amin(float const * __restrict__ b, float const * __restrict__ e) {
 float ret = *(b++);
#pragma omp simd  aligned (b, e : 8 * sizeof (float))
 for(auto k=b;k<e; ++k)
   ret = ret > (*k) ? *k : ret;
 return ret;
}


float const * lmin(float const * __restrict__ b, float const * __restrict__ e) {
 float const * ret = b;

#pragma omp simd
 for(auto k=b;k<e; ++k)
   ret = *ret > (*k) ? k : ret;
 return	ret;
}


float const * lmin2(float const * __restrict__ b, float const * __restrict__ e) {
 float const * ret = b++;
 float amin = *(ret);
 for(;b!=e; ++b)
   if (amin>(*b)) {
     ret = b;
     amin = *ret;
   }
 return ret;
}

float const * lmin3(float const * __restrict__ b, float const * __restrict__ e) {
 float const * __restrict__ p = b;
 float amin = *(p++);
 float n=0;
 for(;p!=e; ++p) {
   n =  (amin>(*p)) ? float(p-b) : n;
   amin = (amin >(*p)) ? *p : amin;
  // if (amin>(*p)) amin = *p;
 }
 return b + int(n+0.5f);
}



int lmin4(float const * __restrict__ c, int N) {
  int k=0;
#pragma omp simd
  for (int i=1; i<N; ++i)
     k =  (c[k] > c[i]) ? i : k;
  return k;
}

int lmin5(float const * __restrict__ c, int N) {
  int  k=0;
  float m = c[0];
  for (int i=1; i!=N; ++i) {
    if (c[i]<m) {
      m=c[i]; k=i;
    }
  }
  return k;
}


float lmin6(float const * __restrict__ c, int * __restrict__ j, int N) {
  j[0]=0;
  float m = c[0];
  for (int i=1; i!=N; ++i) {
    int tmp = j[i];
    if (c[i]<m) tmp=0;
    if (c[i]<m) {
      m=c[i]; 
    }
    j[i]=tmp;
  }
  return m;
}


float fmin(float const * __restrict__ c, int N) {;
  float m = c[0];
  for (int i=1; i!=N; ++i) {
    if (c[i]<m) {
      m=c[i];
    }
  }
  return m;
}


/*
float asum(float const * __restrict__ c, unsigned int N) {
  float ret=0.f;
  for (unsigned int i=0; i!=N; ++i)
    ret += c[i];
  return ret;
}


float sumc(float const * __restrict__ a) {
  float s=0;
  for (int j=0; j!=256; ++j) {
    s += a[j];
  }
  return s;
}


float minc(float * __restrict__ a) {
  float s=a[0];
  for (int j=0; j!=256; ++j) {
    s = s > a[j] ? a[j] : s; 
  }
  return s;
}
*/
