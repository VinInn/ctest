#include <iostream>
#include <vector>


struct Param {
  int j;
  float f;
  std::vector<double> v;

};

class A  : private Param {
public:
  A(const Param& p) : Param(p) {}

  Param const & params() const { return (Param const &)(*this);}


};

class B {
public:
  B(const Param& p) : m_params(p) {}

  Param const & params() const { return m_params;}

private:

  Param const m_params; 

};


struct V {
  static std::vector<double> & vd() {
    static std::vector<double> l;
    return l;
  } 
};

int main() {
  Param p = {1,3.4,V::vd()};
  A a(p);
  B b(p);

  return 0;
}
