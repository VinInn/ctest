#include<iostream>
#include<string>
#include<vector>
#include <iomanip>
#include <cstdlib>
#include <iterator>

int main() {
  { 
    std::vector<unsigned char> s(18,'s');
    std::cout << "size in " << s.size() << " "<< std::endl; 
    for (int j=0; j<int(s.size());j++) std::cout << (unsigned int)(s[j]) << "."; 
    std::cout << std::endl;
    std::cout << "size out " << s.size() << " "<< std::endl; 
    for (int j=0; j<int(s.size());j++) std::cout << (unsigned int)(s[j]) << "."; 
    std::cout << std::endl;
    std::cout << "size out " << s.size() << " "<< std::endl; 

    char a[] = {'a','b','c'};
    std::vector<char> v(3);
    std::copy(a,a+3,v.begin());
    std::copy(v.begin(),v.end(),std::ostream_iterator<char>(std::cout));
    std::cout<< std::endl;
    std::copy(v.rbegin(),v.rend(),std::ostream_iterator<char>(std::cout));
    std::cout<< std::endl;
    std::vector<char>::const_reverse_iterator c = v.rbegin();
    std::cout << *c; ++c;
    std::cout << *c; ++c;
    std::cout<< std::endl;
   
  }
  return 0;
}
