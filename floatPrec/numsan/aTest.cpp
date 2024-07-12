#include<vector>
void foo(double x, std::vector<double> & v);
void sfoo(double x, std::vector<double> & v);

void foo(float x, std::vector<float> & v);
void sfoo(float x, std::vector<float> & v);



#include <iostream>
#include <cmath>


int main(int argc, char **) {

{
  std::vector<double> vd;

  foo(argc,vd);
  sfoo(argc,vd);

  auto x = vd[0];
  auto y = vd[1];

  if (x==y) std::cout << "bizarre" << std::endl;

  std::cout << x << std::endl;
  std::cout << y << std::endl;
  printf("%f %a\n",x,x);

}

{
 std::vector<float> vd;

  foo(float(argc),vd);
  sfoo(float(argc),vd);

  auto x = vd[0];
  auto y = vd[1];

  if (x==y) std::cout << "bizarre" << std::endl;

  std::cout << x << std::endl;
  std::cout << y << std::endl;
  printf("%f %a\n",x,x);



}

  return 0;

}
