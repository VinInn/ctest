inline namespace V2_00 {}

namespace V2_00 {
  struct Hit;
}

namespace V1_00 {
  
  struct Hit {
    Hit(){}
    Hit(int iid, float it): id(iid), time(it){}
    Hit(V2_00::Hit const & newer);
    int id;
    float time;
  };
  
}

namespace V2_00 {
  
  struct Hit {
    Hit(){}
    Hit(int iid, int it, int iq): id(iid), time(it), qual(iq){}
    Hit(V1_00::Hit const & old);
    int id;
    int time;
    int qual;
  };
  
}


V2_00::Hit::Hit(V1_00::Hit const & old) : id(old.id), time(100.f*(old.time)), qual(-1){}
V1_00::Hit::Hit(V2_00::Hit const & newer) : id(newer.id), time(0.01f*float(newer.time)) {}


#include <iostream>
#include <typeinfo>

int main() {
  
  std::cout << typeid(Hit).name() << std::endl;
  std::cout << typeid(V1_00::Hit).name() << std::endl;
  std::cout << typeid(V2_00::Hit).name() << std::endl;

  Hit h;
  std::cout << typeid(h).name() << std::endl;

  V1_00::Hit old(2,3.14);
  Hit curr(old);
  V1_00::Hit back(curr);

  std::cout << old.id << ", " << old.time << std::endl;
  std::cout << curr.id << ", " << curr.time << ", " << curr.qual << std::endl;
  std::cout << back.id << ", " << back.time << std::endl;

}
