#include<tuple>
#include<cmath>
#include<cassert>

struct Base {
  enum Algo { z=0,r=1,eta=2};
  Base(Algo who) : algo(who){}
  virtual float compute(int i) const=0;
  Algo algo;
};


float hr[1024];
float hz[1024];

struct Zalgo final : public Base {
  static constexpr Algo me =z;
  Zalgo() : Base(me){}

  float compute(int i) const { return hz[i]/k; }

  float k;
};

struct Ralgo final : public Base {
  static constexpr Algo me =r;
  Ralgo() : Base(me){}

  float compute(int i) const { return k*std::sqrt(hr[i]); }

  float k;


};

struct Ealgo final : public Base {
  static constexpr Algo me =eta;
  Ealgo() : Base(me){}

  float compute(int i) const { return k*hz[i]/hr[i]; }

  float k;

};

float res[1024];
template<typename Algo>
struct Kernel {

  void set(Base const & a) {
    assert( a.algo==Algo::me);
    algo=reinterpret_cast<Algo const *>(&a);
  }

  void operator()() const {
    for (int i=0; i!=1024; ++i)
      res[i]=(*algo).compute(i);
  }

  Algo const * algo;

};

template<typename ... Args> using Kernels = std::tuple<Kernel<Args>...>;


void go(Base const & a) {

  // original code
  for (int i=0; i!=1024; ++i)
    res[i]=a.compute(i);


  // vectorized
  Kernels<Zalgo,Ralgo,Ealgo> kernels;
  
  switch (a.algo) {
  case (Base::z) :
    std::get<0>(kernels).set(a);
    std::get<0>(kernels)();
    break;
  case (Base::r) :
    std::get<1>(kernels).set(a);
    std::get<1>(kernels)();
    break;
  case (Base::eta) :
    std::get<2>(kernels).set(a);
    std::get<2>(kernels)();
    break;
  }
};
