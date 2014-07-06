unsigned int N;
float * a, *b, *c;

void bar() {
#pragma GCC ivdep
  for (auto i=0U; i<N; ++i)
    a[i] = b[i]*c[i];
}
