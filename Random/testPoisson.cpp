#include "FastPoissonPDF.h"

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <limits>
#include <cmath>
#include <vector>

int main() {
  std::cout << std::setprecision(9); // std::hexfloat;

   float mu = 12.5;
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
