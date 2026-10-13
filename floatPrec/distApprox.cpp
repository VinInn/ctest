#include<cmath>
#include<iostream>
#include<cstdint>
#include<algorithm>

uint32_t dist1( uint16_t dx, uint16_t dy )
{

   auto mx = std::max(dx,dy);
   auto mn = std::min(dx,dy);

   uint32_t approx = ( mx * 1007 ) + ( mn * 441 );
   if ( mx < ( mn << 4 ))
      approx -= ( mx * 40 );

   // add 512 for proper rounding
   return (( approx + 512 ) >> 10 );
} 

uint32_t dist2( uint16_t dx, uint16_t dy )
{

   auto mx = std::max(dx,dy);
   auto mn = std::min(dx,dy);

// coefficients equivalent to ( 123/128 * max ) and ( 51/128 * min )
   return ((( mx << 8 ) + ( mx << 3 ) - ( mx << 4 ) - ( mx << 1 ) +
            ( mn << 7 ) - ( mn << 5 ) + ( mn << 3 ) - ( mn << 1 ) ) >> 8 );

   return ((( mx << 6 ) + ( mx << 6 ) - ( mx << 3 ) - ( mx ) +
            ( mn << 6 ) - ( mn << 4 ) + ( mn << 2 ) - ( mn ) + 64) >> 7 );

} 




#include<random>


int main() {

    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> rint16(0,64*1024); //  65536);

    int n=256*1024*1024;
    int32_t mx1=0, mx2=0;
    double s1=0, s2=0;
    double av1=0, av2=0;
    for (int i=0; i<n; ++i) {
      uint16_t dx = rint16(gen);
      uint16_t dy = rint16(gen);
      int32_t d0 = std::round(std::sqrt(double(dx)*dx+double(dy)*dy));
      int32_t d1 = dist1(dx,dy);
      int32_t d2 = dist2(dx,dy);
      mx1 = std::max(mx1,std::abs(d0-d1));
      mx2 = std::max(mx2,std::abs(d0-d2));
      if (d0>10) {
        s1 = std::max(s1,double(std::abs(d0-d1))/d0);
        s2 = std::max(s2,double(std::abs(d0-d2))/d0);
      }
      av1 += std::abs(d0-d1);
      av2 += std::abs(d0-d2);
    }

   std::cout << mx1 << ' '  << av1/n << ' ' << s1 << std::endl;
   std::cout << mx2 << ' '  << av2/n << ' ' << s2 << std::endl;

}
