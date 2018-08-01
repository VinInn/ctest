#include <iostream>
#include<cassert>
#include<cstdint>

int main() {

  uint32_t cs[10] = {4,7,8,10,20,21,24,55,60,100};

  for (uint32_t i=0; i<200; ++i) {
    uint32_t ind=0;
    while(i>=cs[ind++]); --ind;
    assert(ind<10);
    assert(i<cs[ind]);
    assert (0==ind ||  i>=cs[ind-1]); 
    std::cout << i << ' ' << ind << ' ' << cs[ind] << std::endl;

  }

  return 0;
}
