const int N=1024;
float sum[N];
float diff[N+1];

void adjDiff() {
  for (int i=1; i!=N;++i)
    diff[i]=sum[i]-sum[i-1];
}

void adjDiff1() {
  for (int i=0; i!=N;++i)
    diff[i]-=diff[i+1];
}
