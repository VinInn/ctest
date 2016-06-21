#include <numaif.h>
#include<iostream>


int main() {

    auto p = new int;
    std::cout << get_mempolicy(nullptr,nullptr,64,p,MPOL_F_ADDR) << std::endl;
    delete p;
    std::cout << get_mempolicy(nullptr,nullptr,64,p,MPOL_F_ADDR)  << std::endl;
    p = 0;
    std::cout << get_mempolicy(nullptr,nullptr,64,p,MPOL_F_ADDR)  << std::endl;
    p = (int*)(-123456);
    std::cout << get_mempolicy(nullptr,nullptr,64,p,MPOL_F_ADDR) << std::endl;

  return 0;
}
