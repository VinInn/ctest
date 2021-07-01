__global__ 
void norec(float const * x, float const * y, float const * z, float * q) {
  auto i = threadIdx.x;;
  q[i] = x[i] + y[i]*z[i];
} 


__global__ 
void rec(float const * __restrict__ x, float const * __restrict__ y, float const * __restrict__ z, float * __restrict__ q) {
  auto i = threadIdx.x;;
  q[i] = x[i] +    y[i]*z[i];
}


inline
__device__
void norecf(float const * x, float const * y, float const * z, float * q) {
  auto i = threadIdx.x;;
  q[i] = x[i] + y[i]*z[i];
}

inline
__device__ 
void recf(float const * __restrict__ x, float const * __restrict__ y, float const * __restrict__ z, float * __restrict__ q) {
  auto i = threadIdx.x;;
  q[i] = x[i] +    y[i]*z[i];
}



__global__
void recg(float const * __restrict__ x, float const * __restrict__ y, float const * __restrict__ z, float * __restrict q) {
 norecf(x,y,z,q);
}


__global__
void norecg(float const * x, float const * y, float const * z, float * q) {
  recf(x,y,z,q);
}



struct H {

  float * x;
  float * y;
  float * z;

__device__
__forceinline__
float const * __restrict__ xg() const { return x;}

constexpr
__forceinline__
float const * __restrict__ yg() const { return y;}

__device__ __forceinline__
float zg(int i) const { return __ldg(z+i);}


};



__global__
void rech(H const * __restrict__ ph, float * __restrict__ q) {
  auto const & h = *ph;
  float const * __restrict__ yg = h.y;
  auto i = threadIdx.x;;
  q[i] = h.xg()[i] + yg[i]*h.zg(i);
}


struct AC {

  float const * __restrict__ x;
  float const * __restrict__ y;
  float const * __restrict__ z;
};

__global__
void rechc(AC const h, float * __restrict__ q, float * __restrict__ w) {
  auto i = threadIdx.x;;
  q[i] = h.x[i] + h.y[i]*h.z[i];
  w[i] = h.x[i] - h.y[i]*h.z[i];
}
