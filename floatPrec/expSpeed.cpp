
// performance test
#include<cmath>
#include<iostream>
#include <x86intrin.h>
inline volatile unsigned long long rdtsc() {
 return __rdtsc();
}



int main() {
  union { long long i; double d;} x;
  long long t=0;
  x.d=1.;
  double sum=0;
  t -= rdtsc();
  while(x.d<1.1) {
    x.i+=1000000;
    sum+=std::exp(x.d);
  }
  t += rdtsc();
  std::cout << "time in cycles " << t << std::endl;
 std::cout << "sum= " << sum << " to prevent compiler optim." << std::endl;

 return 0;

}
