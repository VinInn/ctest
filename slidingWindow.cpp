#include<iostream>
#include<limits>
#include<cassert>

int main() {

  uint16_t imax = uint16_t(0) - uint16_t(1);

  // uint16_t mysize = 2356; 
  // uint16_t mymin = 2300;
  // uint16_t mymax = 156;
  std::cout << std::numeric_limits<uint16_t>::max() <<' '<< imax << ' ' << uint16_t(imax+uint16_t(1)) << ' ' << uint16_t(imax+uint16_t(2)) << std::endl;

  auto slidingWindow = [&](uint16_t mysize, uint16_t mymin,uint16_t mymax) {
    uint16_t offset = (mymin>mymax) ? imax-(mysize-1) : 0;
    int n=0;
    for (uint16_t i = mymin+offset; i!=mymax; i++) {
      assert(i<=imax);
      uint16_t k = (i>mymax) ? i-offset : i;
      assert(k<mysize);
      assert(k>=mymin || k<mymax);
      n++;
    }
    int tot = (mymin>mymax) ? (mysize-mymin)+mymax : mymax-mymin;
    assert(n==tot);
  };

  slidingWindow(2356,2300,156);
  slidingWindow(2356,1,500);
  slidingWindow(2356,2355,2300);

  return 0;

}
