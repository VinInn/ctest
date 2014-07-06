constexpr int N=1024;
float a[N];

void acc() {
  for (int i=8; i!=N; ++i) a[i]*=a[i-8];
}
