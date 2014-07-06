#include <iostream>

template<int N> struct Hit;

template<int N>
struct V {

  double v[N];
};

struct HitBase {
  HitBase() : i(-1){}
  virtual ~HitBase(){}
  virtual int dimension() const=0;
  template<int N>
  V<N> const * 
  get() const {
    Hit<N> const * h =
      dynamic_cast<Hit<N> const *>(this);
    if (h) return &(*h).getV();
    return 0;
  }
  
  int i;
};

template<int N> struct Hit : public HitBase {
  virtual int dimension() const { return N;}
  V<N> const & getV() const { return v;}
  
  V<N> v;
};



template< typename T >
void apply(HitBase& hit) {
  T myfunc;
  switch (hit.dimension()) {
  case 1: myfunc.template apply<1>(hit); break;
  case 2: myfunc.template apply<2>(hit); break;
  case 3: myfunc.template apply<3>(hit); break;
  case 4: myfunc.template apply<4>(hit); break;
  case 5: myfunc.template apply<5>(hit); break;
  default: break;
  }
}

struct f {
  template<int N> 
  void apply(HitBase&hit) {
    V<N> const * v = hit. template get<N>();
    if (v==0) hit.i=0;
    else
      hit.i=sizeof((*v))/sizeof(double);
  }
};


int main(int argc) {
  
  Hit<3> h;
  apply<f>(h);
  std::cout << h.i << std::endl;
  if (h.get<2>()) std::cout << "error " << std::endl;

  return 0;
} 
