#include<iostream>

// #Version (CARF)

#define BHA(S) #S

int main() {

#if #Version (CARF)
  std::cout << "CARF " << BHA(Version) << std::endl;
#else
  std::cout << "NO" << BHA(Version) << std::endl;
#endif

  return 0;

}
