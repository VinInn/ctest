#include<iostream>
#include<cmath>
#include<limits>

template<typename T> 
void print(T x) {
 std::cout<< std::hexfloat << x <<' '<<  std::defaultfloat << x << std::endl;
}




int main() {


  float tenth = 0.1;
  print(tenth);
  auto next = std::nextafter(tenth,1.f);
  print(next);
  auto ulp = (next-tenth);
  print (ulp);
  auto prec = ulp/tenth;
  print (prec);

  long long large = 1000*1000*1000;
  std::cout << large << std::endl;
  float flarge = large;
  print(flarge);
  while ( float(++large)==flarge );
  float nlarge = large;
  std::cout << large << std::endl;
  print(nlarge);
  print((nlarge-flarge)/flarge);

  while ( float(++large)==nlarge );
  nlarge = large;
  std::cout << large << std::endl;
  print(nlarge);
  print((nlarge-flarge)/flarge);

  auto nf = [](auto a, auto b) {
    long long n=0;
    while(a!=b) { a=std::nextafter(a,b); ++n;}
    std::cout << n << std::endl;
  };

  nf(0.1f,0.2f); 
  nf(1000.f,2000.f);

  float x1=0;
  for (float y=0;y<=1000000; ++y)x1+=y;
  print (x1);
  float x2=0;
  for (float y=1000000;y>0; --y)x2+=y;
  print (x2);
  print (x2-x1); print ((x2-x1)/x1);

  return 0;

}



