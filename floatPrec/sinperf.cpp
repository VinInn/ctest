#include<cmath>
int main() {

  double res=0;

  double x[1024];
  for (int j=0; j<1024; ++j) x[j] = 0.01*j;

  for (int i=0; i<1024*1024; ++i) {
      auto inc = x[1023];
      for (int j=0; j<1024; ++j) x[j]+=inc;
      for (int j=0; j<1024; ++j) {
         auto y = 1./std::sqrt(std::sqrt(0.0001*x[j]));
         y = std::sin(y);
         res+=y;
      }
  }

  return res > 0.5; 

}
