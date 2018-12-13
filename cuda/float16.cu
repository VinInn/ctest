#include <cuda_fp16.h>
#include <limits>
#include <iostream>



int main() {

  using float16_t = __fp16;

std::cout << "float16\t"
              << float(std::numeric_limits<float16_t>::lowest()) << '\t'
              << float(std::numeric_limits<float16_t>::min()) << '\t'
              << float(std::numeric_limits<float16_t>::max()) << std::endl;


  return 0;

}
