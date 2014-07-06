#include <immintrin.h>
#include <iostream>

template <typename V>
void printV(V v) {
  int N = sizeof(typename V::vtype)/sizeof(typename V::stype);
  for (int i=0; i!=N; ++i)
   std::cout << v.a[i] << " ";
}

template <typename T>
union V4 {
};
template <typename T>
union V8 {    
}; 


template<>
union V4<float> {
  typedef float stype;
  typedef __m128 vtype;
  __m128 v;
  float a[4];
};

template<>  
union V4<double> {
  typedef double stype;
  typedef __m256d vtype;
  vtype v;
  stype	a[4];
};

template<>
union V8<float> {
  typedef float stype;
  typedef __m256 vtype;
  vtype v;
  stype a[8];
};


typedef V4<float> V4F;
typedef	V4<double> V4D;
typedef V8<float> V8F;


int main() {

  V4F vf1; vf1.v = _mm_setr_ps(11,12,13,14);
  V4D vd1; vd1.v = _mm256_setr_pd(11,12,13,14);
  V4F vf2; vf2.v = _mm_setr_ps(21,22,23,24);
  V4D vd2; vd2.v = _mm256_setr_pd(21,22,23,24);
  V4F vf;
  V4D vd;

  printV(vf1); std::cout << std::endl;
  printV(vd1); std::cout << std::endl;
  printV(vf2); std::cout << std::endl;
  printV(vd2); std::cout << std::endl;

  vf.v = _mm_shuffle_ps(vf1.v, vf2.v, _MM_SHUFFLE(1, 0, 3, 2));
  vd.v = _mm256_permute2f128_pd(vd1.v, vd2.v,(2<<4)+1);

  printV(vf); std::cout << std::endl;
  printV(vd); std::cout << std::endl;

  vf.v = _mm_shuffle_ps(vf1.v, vf1.v, _MM_SHUFFLE(2, 3, 0, 1));
  vd.v = _mm256_permute_pd(vd1.v,5);

  printV(vf); std::cout << std::endl;
  printV(vd); std::cout << std::endl;

  vf.v = _mm_shuffle_ps(vf1.v, vf2.v, _MM_SHUFFLE(3, 0, 2, 2));
  vd.v = _mm256_permute2f128_pd(vd1.v, vd2.v,(2<<4)+1);
  vd.v = _mm256_permute_pd(vd.v,0);

  printV(vf); std::cout << std::endl;
  printV(vd); std::cout << std::endl;

  vf.v = _mm_shuffle_ps(vf1.v, vf2.v, _MM_SHUFFLE(3, 1, 0, 1));
  vd.v = _mm256_permute2f128_pd(vd1.v, vd2.v,(2<<4));
  vd.v = _mm256_permute_pd(vd.v,5);

  printV(vf); std::cout << std::endl;
  printV(vd); std::cout << std::endl;
 

  V8F v8f; v8f.v = _mm256_setr_ps(1,2,3,4,5,6,7,8);
  vf.v = _mm256_castps256_ps128(v8f.v);
  printV(v8f); std::cout << std::endl;
  printV(vf); std::cout << std::endl;
  v8f.v =_mm256_permute2f128_ps(v8f.v,v8f.v,1);
  printV(v8f); std::cout << std::endl;

  vf.v = _mm_hadd_ps(vf1.v,vf2.v);
  vd.v = _mm256_hadd_pd(vd1.v,vd2.v);

  printV(vf); std::cout << std::endl;
  printV(vd); std::cout << std::endl;

  vf.v = _mm_hadd_ps(vf.v,vf.v);
  __m256d tmp = _mm256_permute2f128_pd(vd.v,vd.v,1);
  vd.v = _mm256_add_pd(vd.v,tmp);
  
  printV(vf); std::cout << std::endl;
  printV(vd); std::cout << std::endl;



  vf.v =_mm_unpackhi_ps(vf1.v,vf2.v);
  vd.v =_mm256_unpackhi_pd(vd1.v,vd2.v);

  printV(vf); std::cout << std::endl;
  printV(vd); std::cout << std::endl;





  return 0;
}
