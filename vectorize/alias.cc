float mem[3*1024];
void sum() {
  float * a=mem;
  const float * b=mem+1024;
  const float * c=mem+2*1024;
  for (int i=0;i!=1024;++i)
    a[i]=b[i]+c[i];
}

void sumN(int n) {
  float * a=mem;
  /*const*/ float * b=a+n;
  /*const*/ float * c=a+2*n;
  for (int i=0;i!=n;++i)
    a[i]=b[i]+c[i];
}

