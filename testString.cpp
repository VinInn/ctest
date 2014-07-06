#include<string>
#include<iostream>
#include<vector>


// #include<ext/array_allocator.h>
#include "ArrayAllocator.h"

//typedef __gnu_cxx::array_allocator<char, std::vector<char> > FixedSizeAlloc;
typedef vin::array_allocator<char, std::vector<char> > FixedSizeAlloc;
typedef std::basic_string<char,  std::char_traits<char>,  FixedSizeAlloc > String;

int main() {

  const char * ac = "The usual test string";
      typedef size_t     size_type;

      std::vector<char> v1(100);
      std::vector<char> v2(100);

  {  
    std::vector<char> v(100);
    size_type s=0;
    FixedSizeAlloc local_alloc(&v1,&s);
    
    String sa(ac,local_alloc);
    String sb(local_alloc);
    sb = sa;
    String sc(sa,3,4,local_alloc);
    
    std::cout << (long)(ac) << std::endl;
    std::cout << (long)(sa.data()) << std::endl;
    std::cout << (long)(sb.data()) << std::endl;
    std::cout << (long)(sa.data()+3) << std::endl;
    std::cout << (long)(sc.data()) << std::endl;
    std::cout << s << std::endl;

    std::string bha(ac);
    const char * c= bha.c_str();
    std::cout << c << std::endl;
    bha += " more";
    std::cout << c << std::endl;
    const char * d = bha.c_str();
    std::cout << d << std::endl;
 


  }
  std::cout << std::endl;
  {  
    // pool_alloc local_alloc;
    
    std::vector<char> v(100);
    size_type s=0;
    FixedSizeAlloc local_alloc(&v2,&s);
    
    String sa(ac,local_alloc);
    String sb(local_alloc);
    sb = sa;
    String sc(sa,3,4,local_alloc);
    
    std::cout << (long)(ac) << std::endl;
    std::cout << (long)(sa.data()) << std::endl;
    std::cout << (long)(sb.data()) << std::endl;
    std::cout << (long)(sa.data()+3) << std::endl;
    std::cout << (long)(sc.data()) << std::endl;
    std::cout << s << std::endl;

  }

  return 0;
};
