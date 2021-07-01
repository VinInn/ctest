truct H {
  float * __restrict__ x;
  float * __restrict__ y;
  float * __restrict__ z;
};

struct HC {
  float const * __restrict__ x;
  float const * __restrict__ y;
  float const * __restrict__ z;
};


struct K {
  float * x;
  float * y;
  float * z;
};


__global__
void cross(HC v1, HC v2, H v3) {
auto i = threadIdx.x;;
v3.x[i] = v2.y[i]*v1.z[i] +  v1.y[i]* v2.z[i];
v3.y[i] = v2.x[i]*v1.z[i] +  v1.x[i]* v2.z[i];
v3.z[i] = v2.y[i]*v1.x[i] +  v1.y[i]* v2.x[i];
}


__global__
void cross2(K v1, K v2, K v3) {
auto i = threadIdx.x;;
v3.x[i] = v2.y[i]*v1.z[i] +  v1.y[i]* v2.z[i];
v3.y[i] = v2.x[i]*v1.z[i] +  v1.x[i]* v2.z[i];
v3.z[i] = v2.y[i]*v1.x[i] +  v1.y[i]* v2.x[i];
}
