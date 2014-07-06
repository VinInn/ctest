void rsum(float * v, int N) {
  for (int i=0; i!=N; i++)
   v[i]+=v[i-1];
}
