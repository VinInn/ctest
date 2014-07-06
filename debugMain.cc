#include "debug.h"
#include <iostream>


static std::ostream nullOut((std::streambuf*)(0));
#ifdef NDEBUG
# define LOG 0 && nullOut
#else
# define LOG std::cout
#endif

#define OUT doit && std::cout

int main() {

  A a = {3,4.567};

  LOG << MyOut(a).str() << std::endl;

  bool doit = true;
  if (OUT) {
    // complex calculation
    OUT << " YEAH " << std::endl;
  }

  doit = false;

  OUT << "never" << std::endl;
  
  if (OUT) 
    std:: cout << "just to check " << std::endl;

  OUT <<  MyOut(a).str() << std::endl;

  return 0;


}
