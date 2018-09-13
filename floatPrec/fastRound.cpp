#include<cstdint>

template<typename T>
struct FastRoundTrait{
};


template<>
struct FastRoundTrait<float>{
 static constexpr float magic = 12582912.f; // 2^23 +2^22
static constexpr int shift = 10;
 using INT = int32_t;
 union fi { constexpr fi(float q):f(q){} float f; INT i;};
};


template<>
struct FastRoundTrait<double>{
 static constexpr float magic = 6755399441055744.; // 2^52 +2^51
 static constexpr int shift = 13;
 using INT = int64_t;
 union fi { constexpr fi(double q):f(q){}  double f; INT i;};
};


template<typename T>
constexpr typename FastRoundTrait<T>::INT fastRound(T t) {
  typename FastRoundTrait<T>::fi fi(t);
  fi.f +=FastRoundTrait<T>::magic;
  // no need to shift (take lower N/2 bits) if number is small
  return (fi.i<<FastRoundTrait<T>::shift)>>FastRoundTrait<T>::shift;
}






#include<iostream>

#define constexpr const
int main() {


  constexpr int32_t f1 = fastRound(131567.32f);
  constexpr int32_t f2 = fastRound(-131567.32f);
  constexpr int32_t f3 = fastRound(131567.72f);
  constexpr int32_t f4 = fastRound(-131567.72f);

  constexpr auto ff = f3+f4;

  std::cout << f1 << std::endl;
  std::cout << f2 << std::endl;
  std::cout << f3 << std::endl;
  std::cout << f4 << std::endl;
  std::cout << ff << std::endl;

  return 0;

}

 
