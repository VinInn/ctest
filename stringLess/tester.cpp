#include "Config.h"

#include <iostream>


void  Registerer::add(Key obj, Key name, void * conf) {
  std::cout << obj << " " << name << " " << conf << std::endl;
}


int main() {

  return 0;
}
