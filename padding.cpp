#include<iostream>
#include<vector>

int main() {

 struct XC { float fx; char cx;}; XC myxc;
 struct X { float fx;}; X myx;
 struct XXX {float fx; float fy; float fz;}; XXX myxxx;

std::cout << sizeof(myxc) << std::endl;
std::cout <<  sizeof(myx) << std::endl;
std::cout <<  sizeof(myxxx) << std::endl;      
XXX v[10];
std::cout << (long long)(v+10) - (long long)(v) << std::endl;

std::vector<XXX> vv(10);
  std::cout << (long long)(&vv.back()) - (long long)(&vv.front()) << std::endl;



}
