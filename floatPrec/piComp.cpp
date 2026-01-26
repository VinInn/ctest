#include <iostream>
#include <cstdio>

int main() {

  using F = float;

  int N = 100000;
  F step = 1./N;

  F sum=0;
  F x = F(0.5)*step;
  for (int i=0; i<N; ++i)  {
    auto x = (0.5f+F(i))*step;
    sum += F(4.0)/(F(1)+x*x);
    // x+=step;
  }
  printf("%.8f\n",step*sum);
  //std::cout << step*sum << std::endl;  

}
