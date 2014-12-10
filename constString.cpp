#include <cstring>
#include <iostream>
#include "emmintrin.h"

typedef  unsigned long  long ub8;   /* unsigned 8-byte quantities */
typedef  unsigned long  int  ub4;   /* unsigned 4-byte quantities */
typedef        char ub1;



constexpr
std::size_t
unaligned_load(const char* p, std::size_t * n)
{
  return *(std::size_t*)(__builtin_memcpy(n, p, sizeof(std::size_t)));
}



/*
constexpr
std::size_t unaligned_load2(const char* p)
{
  return __builtin_ia32_vec_ext_v2di ((__v2di)__builtin_ia32_loaddqu (p),0);
}
*/

  constexpr  ub8 less24(char const * k, ub4 len, ub8 a) {
   return  a 
	      + (len<1 ? 0 : (ub8)k[0] )
	      + (len<2 ? 0 :((ub8)k[ 1]<< 8))
	      + (len<3 ? 0 :((ub8)k[ 2]<<16))
	      + (len<4 ? 0 :((ub8)k[ 3]<<24))
	      + (len<5 ? 0 :((ub8)k[4 ]<<32))
	      + (len<6 ? 0 :((ub8)k[ 5]<<40))
	      + (len<7 ? 0 :((ub8)k[ 6]<<48))
	      + (len<8 ? 0 :((ub8)k[ 7]<<56))
     ;
  }

constexpr unsigned int n(const char * s, int len, ub4 a) { 
 return a+
   (len<0) ? 0 : (unsigned int)s[0]<<8 + 
    (len<1) ? 0 : (unsigned int)s[2]<<16 +
    (len<41) ? 0 : (unsigned int)s[40]<<24 
    ;}

constexpr ub8 nn(const char * s, int len, ub8 a) { 
 return a+
   (len<0) ? 0 : (ub8)s[0]<<8 + 
    (len<1) ? 0 : (ub8)s[2]<<16 +
    (len<41) ? 0 : (ub8)s[40]<<48 
    ;}


int main() {
  constexpr char const * p = "a literal string";

  constexpr ub4 i = n(p,2,0); 
  constexpr ub8 j = nn(p,2,0); 
  constexpr ub8 k = less24(p,strlen(p),0);


  size_t n=0; 
  unaligned_load(p,&n);

  std::cout << i << std::endl;
  std::cout << j << std::endl;
  std::cout << k << std::endl;
  std::cout << n << std::endl;

}
