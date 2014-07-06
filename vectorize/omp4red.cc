float a[1024];
float b[1024];

float sumO1() {
 float s = 0.f;
#pragma omp simd reduction(+:s)
  for (int i=0;i<1024;++i) {
    s += a[i]*b[i];
  }
  return s;
}
