float __attribute__((__target__("sse2", "sse4","avx"))) foo (float const * __restrict__ x, float const * __restrict__ y, int N ) {
   float sum=0.f;
   for (int i=0; i!=N; ++i)
	sum += x[i]*y[i];
  return sum;
}
