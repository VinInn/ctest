#include <cmath>
#include <iostream>
#include <cstdio>

int main() {


    std::cout << "on CPU" <<  std::endl;
    volatile float x = 0x1.fffffcp+1; // 0x1.f02102p-13;
    // volatile float x = 0x1.f02102p-13;

    volatile float y = 1.f/sqrtf(x);
    volatile float z = rsqrtf(x);
    volatile float w = 1.f/std::sqrt(x);
    volatile float d = 1./sqrt(double(x));
    volatile float e = rsqrt(double(x));
    printf ("rsqrt(%a) = %a  %a %a %a %a\n", x, y, z, w, d, e);

}
