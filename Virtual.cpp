#include <cmath>

struct Base {
  virtual ~Base(){}
  virtual double comp() const=0;
};

#ifndef FINAL
#define final
#endif

struct A final : public Base {
  A(){}
  explicit A(double ix) : x(ix){}
  ~A(){}
   double comp() const override { return std::sqrt(x);}
   
  double x;
};


#include<vector>
#include<memory>

int main() {

  int size=1000*10;

  std::vector<A> va(size,A(3.14));
  std::vector<A const *> pa; pa.reserve(size);
  for (auto const & a : va) pa.push_back(&a);

  double c=0;
  for (int i=0; i<10000; ++i) {
    for (auto const & p : pa) c += p->comp();
  }

 return int(c);

}


