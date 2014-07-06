typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef int   __attribute__( ( vector_size( 16 ) ) ) int32x4_t;


float32x4_t shuffle1(float32x4_t x) {
   return float32x4_t{x[1],x[0],x[3],x[2]};
}

 
float32x4_t shuffle2(float32x4_t const & x) {
   return float32x4_t{x[1],x[0],x[3],x[2]};
}

float32x4_t shuffle3(float32x4_t const & x) {
   return __builtin_shuffle(x,int32x4_t{1,0,3,2});
}

struct foo {
  float32x4_t x;
  float32x4_t shuffle2() const;
  float32x4_t shuffle3() const;
};

float32x4_t foo::shuffle2() const {
  return float32x4_t{x[1],x[0],x[3],x[2]};
}
float32x4_t foo::shuffle3() const {
   return __builtin_shuffle(x,int32x4_t{1,0,3,2});
}

