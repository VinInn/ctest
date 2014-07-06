#include<iostream>
#include<limits>


int main() {

#ifdef __CHAR_UNSIGNED__
  std::cout << "CHAR_UNSIGNED" << std::endl;
#else
  std::cout << "Not CHAR_UNSIGNED" << std::endl;
#endif
   std::cout << "Checking " << "unsigned char"
             << "; min is " << CHAR_MIN
	     << ", max is " << CHAR_MAX
	     << std::endl;
   
   std::cout << "limit" <<std::endl;
   std::cout << "Checking " << "signed char"
	     << "; min is " << long(std::numeric_limits<unsigned char>::min())
	     << ", max is " << long(std::numeric_limits<unsigned char>::max())
	     << std::endl;


   printf("%d %x\n",std::numeric_limits<unsigned int>::max());

  return 0;
}
