#include <random>
#include <iostream>
#include <iomanip>
#include <ios>
#include "canonical.h"
#include "luxFloat.h"

#include<algorithm>
int main () {
  std::cout << std::setprecision(9); // std::hexfloat;
  constexpr canonical_float_random<float, std::mt19937> canonical_dist;
  std::mt19937 gen1;
  std::mt19937 gen2;
  std::cout << gen1() << ' ' << gen2() << std::endl;
  std::cout << canonical_dist (gen1) << std::endl;
  std::cout << luxFloat(gen2) << std::endl;
  
  int N = 1000 * 1000 * 1000;
  float mn[2] = {2.,2,};
  float mx[2] = {-2.,-2.};
  double av[2]={0,0};
  for (int i=0; i<N; ++i) {
    auto f1 = canonical_dist (gen1);
    auto f2 = luxFloat(gen2);
    av[0] +=f1;
    av[1] +=f2;
    mn[0] = std::min(mn[0],f1);
    mx[0] = std::max(mx[0],f1);
    mn[1] = std::min(mn[1],f2);
    mx[1] = std::max(mx[1],f2);
  }

  std::cout << mn[0] << ' ' << mx[0] << ' ' << av[0]/N << std::endl;
  std::cout << mn[1] << ' ' << mx[1] << ' ' << av[1]/N << std::endl;

  return 0;
}
