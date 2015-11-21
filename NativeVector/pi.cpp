#include <limits>
#include <algorithm>
#include <type_traits>
#include <cstring>
#include <x86intrin.h>


// c++ -Ofast pi.cpp  -fopt-info-vec -march=native

typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;
constexpr float32x4_t zero4= {0.f,0.f,0.f,0.f};

template<typename Float>
float pi(int num_steps) {
  if(num_steps <=0) __builtin_unreachable();
  Float step =  Float(1.0)/(Float) num_steps;
  Float sum = 0;
  // num_steps = 4*(num_steps/4);  // make sure is a multiple of 4
  for (int i=0;i< num_steps; i++){
    auto x = (Float(i)+Float(0.5))*step;
    sum += Float(4.0)/(Float(1.0)+x*x);
  }
  return step * sum;

}

template<>
float pi<float32x4_t>(int num_steps) {
  if(num_steps <=0) __builtin_unreachable();
  constexpr float32x4_t incr{0.5f,1.5f,2.5f,3.5f};
  using Float = float;
  num_steps = 4*(num_steps/4);  // make sure is a multiple of 4
  Float step =  Float(1.0)/(Float)num_steps;
  float32x4_t sum = zero4;
  for (int i=0;i< num_steps; i+=4){
    auto x = (incr+Float(i))*step;
    sum += Float(4.0)/(Float(1.0)+x*x);
  }
  return step * (sum[0]+sum[1]+sum[2]+sum[3]);

}

#include<iostream>
#include <chrono>

template<typename T>
void go(int num_steps) {
  auto start = std::chrono::high_resolution_clock::now();
  auto res = pi<T>(num_steps);
  auto total_time = std::chrono::high_resolution_clock::now() -start;
  std::cout << "pi = " << res << " in " << total_time.count() << std::endl;
}

int main ()
{

  auto total_time = std::chrono::high_resolution_clock::duration{};

  
  constexpr int num_steps = 1000000;
  
  go<float>(num_steps);
  go<double>(num_steps);
  go<float32x4_t>(num_steps);

  return 0;
}
