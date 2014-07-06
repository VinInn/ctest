#include "debug.h"
#include <iostream>

   
void MyOut::format() { 
  std::cerr << "in format" << std::endl;
  os << m_a.a <<", " << m_a.b; 
}
