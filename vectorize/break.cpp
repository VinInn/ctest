float x[1024];
float y[1024];
float k[1024];

int foo() {
  int i=0;
  int j=0;
  for (; i!=1024; ++i)
    // if (x[i]!=y[i]) k[i]=1.f;
    k[i] = x[i]!=y[i] ? 1.f : k[i];
  return j;
}
