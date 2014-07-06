#include <iostream>

enum Type { valid = 0, missing = 1, inactive = 2, bad = 3 };

inline
Type k(int i) {
  return Type(i);

}

inline
bool isValid(int i) {
  return k(i)==valid;
}

int main() {

  {
    Type t=valid;
    
    int i = t;
    
    std::cout << t << std::endl;
    std::cout << i << std::endl;
    std::cout << k(i) << std::endl;
    std::cout << isValid(i) << std::endl;
    std::cout << std::endl;
  }

  {
    Type t=missing;
    
    int i = t;
    
    std::cout << t << std::endl;
    std::cout << i << std::endl;
    std::cout << k(i) << std::endl;
    std::cout << isValid(i) << std::endl;
  }


}
