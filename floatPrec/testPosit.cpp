#include <universal/number/posit/posit.hpp>


#include<iostream>

template<int N> 
void doit() {
  constexpr int NB = 21;
  std::cout << "posit" << NB << ','<< N << std::endl;
  double mul  = 5;
  for (double  x = 1.e-6; x<1.e6; x*=mul) {
    sw::universal::posit<NB,N> p = x; 
    auto u = p.encoding();
    auto f = x;
    for(;;) {
      f*= 0.75;
      auto y = x - f;
      sw::universal::posit<NB,N> py = y;
      auto uy = py.encoding();
      if (u==uy) {
        std::cout << "err " << x  << ' ' << std::abs(y-x)  << ' ' << std::abs(y-x)/x << std::endl;
        std::cout << x << ' ' << y << ' ' << std::abs(y-x) << ' ' << y/x << ' ' << std::abs(y-x)/x << ' ' << p << ' ' <<  py << ' ' << u << ' ' << uy << std::endl;
        break;
      }
    }
    mul = mul==5 ? 2: 5;
  }
  std::cout << std::endl;
}


int main() {

  doit<0>();
  doit<1>();
  doit<2>();
  doit<3>();
  doit<4>();


  return 0;
}




