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

  std::vector<std::unique_ptr<A>> va(size);

  for (auto & a : va) a = std::make_unique<A>(3.14);

  double c=0;
  for (int i=0; i<10000; ++i) {
    for (auto const & a : va) c += a->comp();
  }

 return int(c);

}


