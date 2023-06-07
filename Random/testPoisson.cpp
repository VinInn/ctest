#include "FastPoissonPDF.h"


#include <random>
#include "Xoshiro.h"

// using Generator = std::mt19937_64;
using Generator = XoshiroPP;


#include <iostream>
#include <iomanip>
#include <ios>

#include <atomic>
#include <thread>
#include <mutex>


#include <cstdint>
#include <limits>
#include <cmath>
#include <vector>


union UInt {
  uint64_t i64;
  uint32_t i32[2];
  uint16_t i16[6];
  uint16_t i8[16];
}

void go(float mu) {

  std::mutex histoLock;
  int h32[40];
  int h21[40];
  int h16[40];
  for (int i=0; i<40; ++i)
      h32[i]=h21[i]=h16[i]=0;

  std::atomic<int> seed=0;;
  std::atomic<long long> iter = 0;
  int64_t N = 1000LL;
  if (argc>1) N *= 1000LL;
  auto run = [&]() {
    seed+=1;
    Generator gen(seed);
    int lh32[40];
    int lh21[40];
    int lh16[40];
    for (int i=0; i<40; ++i)
      lh32[i]=lh21[i]=lh16[i]=0;
    FastPoissonPDF<32> pdf32(mu);
    FastPoissonPDF<21> pdf21(mu);
    FastPoissonPDF<16> pdf16(mu);

    while (iter++ < N) {
    std::cout << '.';
    for (int64_t k=0; k<10000; ++k) {
     for (int64_t i=0; i<64; ++i){
      UInt r1, r2;
      auto r1.i64 = gen();
      auto r2.i64 = gen();
      lh32[std::clamp(0,40,pdf32(r1.i32[0])]++;
      lh32[std::clamp(0,40,pdf32(r1.i32[1])]++;
      lh32[std::clamp(0,40,pdf32(r2.i32[0])]++;
      lh32[std::clamp(0,40,pdf32(r2.i32[1])]++;
      for (j=0; j<4; ++j) 
        lh16[std::clamp(0,40,pdf16(r1.i16[i])]++;
     }
    }
    } // while
    std::cout << std::endl;

  };

}


int main(int argc, char** argv ) {
  std::cout << std::setprecision(9); // std::hexfloat;

   float mu = 12.5;
   if (argc>1) mu = ::atof(argv[1]);
   std::cout << "mu " << mu << std::endl;

   {
   std::cout << "float" << std::endl;
   std::vector<float> cumulative;   
   auto poiss = std::exp(-mu);
   cumulative.push_back(poiss);
   bool zero=true;
   for (;;) {
     poiss *= mu / float(cumulative.size());
     if (zero && poiss > std::numeric_limits<float>::epsilon()) zero=false;
     if ((!zero) && poiss <= std::numeric_limits<float>::epsilon())
          break;
     auto val = cumulative.back() + poiss;
     if (val >= 1.f)
          break;
    cumulative.push_back(val);
   }
   std::cout << cumulative.size()  << std::endl;
   for (auto v:cumulative) std::cout << v << ' ' ;
   std::cout << std::endl;
   }
   {
   std::cout << "uint16" << std::endl;
   std::vector<uint16_t> cumulative;
   double poiss = std::exp(-mu);
   double sum = poiss;
   double mul = std::numeric_limits<uint16_t>::max();
   cumulative.push_back(mul*sum+0.5);
   for (;;) {
     poiss *= mu / cumulative.size();
     sum += poiss;
     if (mul*sum+0.5 >= std::numeric_limits<uint16_t>::max())
          break;
     cumulative.push_back(mul*sum+0.5);
   }
   std::cout << cumulative.size()  << std::endl;
   for (auto v:cumulative) std::cout << v << ' ' ;
   std::cout << std::endl;
   }


   {
   std::cout << "uint32" << std::endl;
   std::vector<uint32_t> cumulative;
   double poiss = std::exp(double(-mu));
   double sum = poiss;
   double mul = std::numeric_limits<uint32_t>::max();
   cumulative.push_back(mul*sum+0.5);
   for (;;) {
     poiss *= double(mu) / cumulative.size();
     sum += poiss;
     // if (sum>=1. -std::numeric_limits<float>::epsilon() ) break;
     if (mul*sum+0.5 >= std::numeric_limits<uint32_t>::max())
          break;
     cumulative.push_back(mul*sum+0.5);
   }
   std::cout << cumulative.size()  << std::endl;
   for (auto v:cumulative) std::cout << v << ' ' ;
   std::cout << std::endl;
   }



   {
   std::cout << "32 bits" << std::endl;
   FastPoissonPDF pdf(mu);   
   std::cout << pdf.cumulative().size()  << std::endl;
   for (auto v:pdf.cumulative()) std::cout << v << ' ' ;
   std::cout << std::endl;
   }


   {
   std::cout << "24 bits" << std::endl;
   FastPoissonPDF<24> pdf(mu);
   std::cout << pdf.cumulative().size()  << std::endl;
   for (auto v:pdf.cumulative()) std::cout << v << ' ' ;
   std::cout << std::endl;
   }
   {
   std::cout << "16 bits" << std::endl;
   FastPoissonPDF<16> pdf(mu);
   std::cout << pdf.cumulative().size()  << std::endl;
   for (auto v:pdf.cumulative()) std::cout << v << ' ' ;
   std::cout << std::endl;
   }

return 0;
}
