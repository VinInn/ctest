#include <iostream>

int pow10(int n) {
 if (n==0) return 1;
 int ten=10;
while (--n>0) ten*=10;
return ten;
}


int main() {
    unsigned int flags_ = -1;
    uint32_t rawEnergy = (0x1FFF & flags_>>4);
    uint16_t exponent = rawEnergy>>10;
    std::cout << exponent << std::endl;
    std::cout << ((0x1FFF)>>10) << std::endl;

std::cout << pow10(0) << std::endl;
std::cout << pow10(1) << std::endl;
std::cout << pow10(3) << std::endl;

}

