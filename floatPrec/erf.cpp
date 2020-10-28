#include<cmath>

float gint(float x) {

  constexpr auto a0 = 1.f/2.f;
  constexpr auto a1 = 1.f/4.f;
  constexpr auto a2 = 1.f/18.f;
  constexpr auto a3 = 1.f/144.f;
  constexpr auto a4 = 1.f/2016.f;
  constexpr auto a5 = 1.f/60480.f;

  constexpr auto b2 = 1.f/9.f;
  constexpr auto b4 = 1.f/10008.f;

  auto p = a0 + x*(a1+x*(a2+x*(a3+x*(a4+x*a5))));
  auto x2 = x*x;
  auto q = 1.f+ x2*(b2+x2*b4);
  return p/q;
}


float  approx_erfc(float y) {

  auto x = std::abs(y);

  constexpr auto a1 = 0.278393f; 
  constexpr auto a2 = 0.230389;
  constexpr auto a3 = 0.000972;
  constexpr auto a4 = 0.078108;
  auto p = 1.f + x*(a1+x*(a2+x*(a3+x*a4)));
  p *=p;
  p *=p;
  return y>0 ? 1.f/p : 2.f - 1.f/p;

}


#include<iostream>
int main() {


   for (float y=-4.; y<4; y+=0.1) {
     std::cout << 1.f - 0.5f*std::erfc(y/std::sqrt(2.f)) << ' ';
   }
    std::cout << std::endl;
    std::cout << std::endl;

   for (float y=-4.; y<4; y+=0.1) {
     std::cout << gint(y) << ' ';
   }
    std::cout << std::endl;
   std::cout << std::endl;

   for (float y=-4.; y<4; y+=0.1) {
     std::cout  << 1.f - 0.5f*std::erfc(y/std::sqrt(2.f))- gint(y) << ' ';
   }
    std::cout << std::endl;   std::cout << std::endl;

   for (float y=-4.; y<4; y+=0.1) {
     std::cout << 1.f - 0.5f*approx_erfc(y/std::sqrt(2.f)) << ' ';
   }

    std::cout << std::endl;
    std::cout << std::endl;

  for (float y=-4.; y<4; y+=0.1) {
     std::cout  << 0.5f*std::erfc(y/std::sqrt(2.f)) - 0.5f*approx_erfc(y/std::sqrt(2.f)) << ' ';
   }
    std::cout << std::endl;   std::cout << std::endl;


  return 0;



}
