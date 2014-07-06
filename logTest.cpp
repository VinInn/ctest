#include "icsiLog.h"
#include <iostream>


int main() {

float x = 2.3;

std::cout << std::log(x) << " " << log16(x) << std::endl;
std::cout << std::exp(std::log(x)) << " " << std::exp(log16(x)) << std::endl;

return 0;
}
