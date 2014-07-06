#include <sstream>

struct A {
  int a;
  float b;
};

struct MyOut {
  MyOut(const A& ia): m_a(ia){}
  std::string str() {
    format();
    return os.str();
  }
   
  void format();

  const A& m_a;
  std::ostringstream os;
};
