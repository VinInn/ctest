#include<iostream>
#include<typeinfo>

int main() {

unsigned char  data[16];
for (auto & c : data) c=1;

std::cout << typeid(data[0]&0x03).name() << std::endl;

for (int i=0; i<16; i+=2) {
   unsigned short v = 
     data[i] + ( (data[i+1]&0x03)<<8);
   std::cout << v << std::endl;
}

for (auto i=0U; i<0X1000U; ++i) 
    std::cout << i<<','<<(i^7)<<' ';
    std::cout << std::endl;


}
