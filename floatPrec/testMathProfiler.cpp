#include<cmath>



int main() {
  int ret = 0;

  for (int i =-33; i<33; ++i) {
    ret += std::sin(float(i));
    ret += std::sin(double(i));
  }
  double x = std::pow(2,-129);
  for (int i =0; i<258; ++i) { 
    x*=2;
    ret += std::asin(float(x));
    ret += std::asin(double(x));
  }

  return ret;
}
