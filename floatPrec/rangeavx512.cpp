#include <x86intrin.h>
#include <iostream>

float amin(float const & a, float const & b) {
 float r;
 _mm_store_ss( &r, _mm_range_ss(_mm_load_ss(&a),_mm_load_ss(&b),14));
  return r;
}

float amax(float const & a, float const & b) {
 float r;
 _mm_store_ss( &r, _mm_range_ss(_mm_load_ss(&a),_mm_load_ss(&b),15));
  return r;
}


int main() {
  std::cout << amin(-10., -5.) << std::endl;
  std::cout << amax(-10., -5.) << std::endl;
  std::cout << amin(-10., 5.) << std::endl;
  std::cout << amax(10., -20.) << std::endl;

  return 0;
}

