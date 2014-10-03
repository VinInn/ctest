#include<iostream>

int main() {

  unsigned int uj = 1;
  int sj = 1;
  unsigned int ui = sj;
  int si = uj;

  std::cout << ui+si << std::endl;

  float x[3], y[3];
  for (unsigned int i=0U; i<3U; ++i)  { x[i] = -i*10; y[i] = -10*int(i);}
  std::cout << x[2] << ' ' << y[2] << std::endl;

  return 0;
}
