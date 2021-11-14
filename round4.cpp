#include<cmath>
#include<iostream>
#include<algorithm>
#include<cassert>
int main() {

  const int N = 6;
  for (int i=N; i<20; ++i) {
    float incr = std::max(float(i)/float(N),1.f);
    float  n=0;
    std::cout << i << " trunc: ";
    for (int j=0; j<N; j++) {  
      std::cout<< int(n) << ',';
      assert(int(n)<i);
      n +=incr;
    }
    std::cout << std::endl;
    n=0;
    std::cout << i << " round: ";
    for (int j=0; j<N; j++) {
      std::cout<< int(n+0.5f) << ',';
      assert(int(n)<i);
      n +=incr;
    }
    std::cout << std::endl;

  }

  return 0;
}
