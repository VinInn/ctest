typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
typedef float __attribute__( ( vector_size( 16 ) , aligned(4) ) ) float32x4a4_t;


float32x4_t load(float32x4a4_t x) {
   return x;
}

// float v[3*1024];

float32x4_t load(float const * x) {
   return *(float32x4a4_t const *)(x);
}

void store(float * x, float32x4_t v) {
   *(float32x4a4_t*)(x) = v;
}


void add(float * x, float32x4_t v) {
   *(float32x4a4_t*)(x) += v;
}


void add11(float * x, float * y, float32x4_t v) {
   auto & k1 = *(float32x4a4_t*)(x);
   auto & k2 = *(float32x4a4_t*)(y);
   k1 +=v;
   k2 += k1+v;
}

void add14(float * x, float * y, float32x4_t v) {
   decltype(auto) k1 = *(float32x4a4_t*)(x);
   decltype(auto) k2 = *(float32x4a4_t*)(y);
   k1 +=v;
   k2 += k1+v;
}


void add98(float * x, float * y, float32x4_t v) {
   float32x4a4_t & k1 = *(float32x4a4_t*)(x);
   float32x4a4_t & k2 = *(float32x4a4_t*)(y);
   k1 +=v;
   k2 += k1+v;
}




int main() {

  float v[3*1025]{1};

  float32x4_t v1{1,2,3,4};

  float32x4_t v2 = load(v+3);
  
  for (int i=0;i<1024; i+=3)
    store(v+i,v1);

  for (int i=0;i<1024; i+=3)
    add(v+i,v1);


 return v[124];
};
