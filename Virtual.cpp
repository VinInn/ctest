#include <cmath>

struct Base {
  virtual ~Base(){}
  virtual double comp() const=0;
};

#ifndef MAKEF
#define FINAL final
#else
#define FINAL
#endif

struct A FINAL : public Base {
  A(){}
  explicit A(double ix) : x(ix){}
  ~A(){}
   double comp() const override { return std::sqrt(x);}
   
  double x;
};

struct B FINAL : public Base {
  B(){}
  explicit B(double ix) : x(ix){}
  ~B(){}
   double comp() const override { return std::sqrt(x);}

  double x;
};


#include<vector>
#include<memory>
#include<random>
#include<algorithm>

int main() {


  int size=1000*10;

  std::vector<A> va(size,A(3.14));
  std::vector<B> vb(size,B(7.1));
  std::vector<Base const *> pa; pa.reserve(2*size);
  int i=0; for (auto const & a : va) { pa.push_back(&a); pa.push_back(&vb[i++]); }
  std::random_shuffle(pa.begin(),pa.end());  

  double c=0;
  for (int i=0; i<10000; ++i) {
    for (auto const & p : pa) c += p->comp();
  }

 return int(c);

}


