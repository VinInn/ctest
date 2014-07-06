void loop(float *  x, int n) {
  for (int i=0;i!=n; ++i)
    x[i]=x[i+1+n]+x[i+1+2*n];
}


float mem[4096];
const int N=1024;

struct XYZ {
  float * __restrict__ mem;
  int n;
  float * __restrict__ x() { return mem;}
  float * __restrict__ y() { return x()+n;}
  float * __restrict__ z() { return y()+n;}
};

inline
void sum(float * x, float * y, float * z, int n) {
  for (int i=0;i!=n; ++i)
    x[i]=y[i]+z[i];
}

void sum2() {
  sum(mem,mem+N,mem+2*N,N);
}

void sumN(int n) {
  sum(mem,mem+n,mem+2*n,n);
}




void sumS() {
  XYZ xyz; xyz.mem=mem; xyz.n=N;
  sum(xyz.x(),xyz.y(),xyz.z(),xyz.n);
}

void sumS2(XYZ & xyz) {
  sum(xyz.x(),xyz.y(),xyz.z(),xyz.n);
}


/*
void foo() {
  for (int i=0;i!=N; ++i)
    mem[i]=mem[N+i]+mem[2*N+i];
}

inline
void bar(float * x, int n) {
  for (int i=0;i!=n; ++i)
    x[i]=x[n+i]+x[2*n+i];
}

void sbar() {
  bar(mem,N);
}
*/
