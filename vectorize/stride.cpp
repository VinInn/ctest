#include<cassert>

template<typename T>
inline size_t slowstride(size_t size, size_t ALIGNMENT) {
    auto off = (size*sizeof(T))%ALIGNMENT;    
    return (off==0) ? size : (ALIGNMENT/sizeof(T))*(size/(ALIGNMENT/sizeof(T))+1);
} 

template<typename T>
inline size_t stride(size_t size, size_t ALIGNMENT) {
    auto mask = ALIGNMENT/sizeof(T)-1;
    return (size&mask) ? (size|mask) + 1 : size;
}



#include<iostream>

int main() {


    for (auto al=16;al!=256; al*=2) {
      std::cout << "\nAL " << al << " " << al/sizeof(double) << " " << al/sizeof(float) << "\n" << std::endl;
      for (auto s=0; s!=64000; ++s) {
//        std::cout << s << " " << slowstride<double>(s,al) << " " << stride<double>(s,al) << std::endl;
//        std::cout << s << " " << slowstride<float>(s,al) << " " << stride<float>(s,al) << std::endl;
        assert(slowstride<double>(s,al)==stride<double>(s,al));
        assert(slowstride<float>(s,al)==stride<float>(s,al));
      }
    }
    return 0;
}
