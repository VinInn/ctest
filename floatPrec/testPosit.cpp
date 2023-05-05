#include <universal/number/posit/posit.hpp>


#include<iostream>

template<int N> 
void doit() {
  std::cout << "posit 24 " << N << std::endl;
  for (double  x = 1.e-6; x<1.e6; x*=10) {
    sw::universal::posit<24,N> p = x; 
    auto u = p.encoding();
    auto f = x;
    for(;;) {
      f*= 0.5;
      auto y = x - f;
      sw::universal::posit<24,N> py = y;
      auto uy = py.encoding();
      if (u==uy) {
        std::cout << "err " << x  << ' ' << std::abs(y-x)  << ' ' << std::abs(y-x)/x << std::endl;
        std::cout << x << ' ' << y << ' ' << std::abs(y-x) << ' ' << y/x << ' ' << std::abs(y-x)/x << ' ' << p << ' ' <<  py << ' ' << u << ' ' << uy << std::endl;
        break;
      }
    }
  }
  std::cout << std::endl;
}


int main() {

  doit<0>();
  doit<1>();
  doit<2>();
  doit<3>();


  return 0;
}




