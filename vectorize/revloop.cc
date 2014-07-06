float x[1024];
float y[1024];
float z[1024];

void foo() {
  for (int i=0; i<512; ++i)
    x[1023-i] += y[1023-i]*z[512-i];
}

void foo2() {
  for (int i=511; i>=0; --i)
    x[1023-i] += y[1023-i]*z[512-i];
}


void bar() {
  for (int i=0; i<512; ++i)
    x[1023-i] += y[i]*z[i+512];
}

void bar2() {
  for (int i=511; i>=0; --i)
    x[1023-i] += y[i]*z[i+512];
}



/*
void bar2(float * x, float * y, float * z, int n1, int n2) {
#pragma GCC ivdep
  for (int i=n1; i<n2; ++i)
    x[n2-i] += y[i]*z[i+n1];
}
*/
