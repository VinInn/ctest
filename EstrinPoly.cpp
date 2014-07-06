#include<algorithm>

template<typename T, int N>
class HornerPoly {
public:
  HornerPoly(){}
  HornerPoly(std::initializer_list<T> il) : p(std::begin(il)+1), c0(*il.begin()){}
  HornerPoly(T const coeff[N+1]) : p(coeff+1), c0(*(coeff)){};
  T operator()(T x) const { return c0 + x*p(x); }
private:
  HornerPoly<T,N-1> p;
  T c0; 
};

template<typename T>
class HornerPoly<T,0> {
public:
  HornerPoly(){}
  HornerPoly(T coeff) : c0(coeff){};
  HornerPoly(T const * coeff) : c0(*coeff){};
  T operator()(T) const { return c0; }
private:
  T c0; 
};


namespace padeDetails {
  template<typename T>
  struct Value {
    Value(){}
    Value(T ip, T iq): p(ip),q(iq){}
    void prod(Value<T> & c, T x)  const {c.p=c.p*x+p; c.q=c.q*x+q;}
    T div() const {return p/q;}

    T p,q;
  };

}

template<typename T, int N>
class PadePoly {
public:
  using ValueType=padeDetails::Value<T>;
  PadePoly(){}
  PadePoly(std::initializer_list<T> il) : pp(std::begin(il)+2), c0(*il.begin(),*(il.begin()+1)){}
  PadePoly(T const coeff[2*(N+1)]) : pp(coeff+2), c0(*(coeff),*(coeff+1)) {};

  void value(ValueType & c,T x) const { pp.value(c,x); c0.prod(c,x) ;}
  // T operator()(T x) const { ValueType v; value(v,x); return v.div(); }

  T p (T x) const { return c0.p + x*pp.p(x); }
  T q (T x) const { return c0.q + x*pp.q(x); }
  T operator()(T x) const { return p(x)/q(x); }

private:
  PadePoly<T,N-1> pp;
  ValueType c0;
};

template<typename T>
class PadePoly<T,0> {
public:
  using ValueType=padeDetails::Value<T>;
  PadePoly(){}
  PadePoly(std::initializer_list<T> il) : c0(*il.begin(),*(il.begin()+1)){}
  PadePoly(T const * coeff) : c0(*coeff,*(coeff+1)){};
  void value(ValueType & c, T) const { c=c0;}
  T operator()(T) const { return c0.div(); }

  T p(T) const { return c0.p; }
  T q(T) const { return c0.q; }

private:
  ValueType c0; 
};





namespace estrinDetails {
  constexpr int ilog2(int i, int n=0) { return i<=1 ? n : ilog2(i/2,n+1); };
  template<typename T, int Q>
  inline T xq(T x) { return xq<T,(Q/2)>(x)*xq<T,(Q/2)>(x); }
  
  template<>
  float xq<float,1>(float x) { return x; }
  template<>
  double xq<double,1>(double x) { return x; }
}

template<typename T, int N>
class EstrinPoly {
private:
  static constexpr int K = estrinDetails::ilog2(N);
  static constexpr int M = ((N>>K)<<K)-1;
public:
  EstrinPoly(){}
  EstrinPoly(std::initializer_list<T> il) : p1(il),p2(il.begin()+M+1)  {}
  EstrinPoly(T const coeff[N+1]) : p1(coeff), p2(coeff+M+1){};
  T operator()(T x) const { T xx = estrinDetails::xq<T,M>(x*x); return p1(x) + xx*p2(x);}
private:;
  EstrinPoly<T,M> p1;
  EstrinPoly<T,N-M-1> p2;
};


template<typename T>
class EstrinPoly<T,0> {
public:
  EstrinPoly(){}
  EstrinPoly(T const * c0) : coeff(*c0){}
  T operator()(T) const { return coeff;}
private:
  T coeff;
};

template<typename T>
class EstrinPoly<T,1> {
public:
  EstrinPoly(){}
  EstrinPoly(std::initializer_list<T> il) : coeff{*il.begin(),*(il.begin()+1)}{}
  EstrinPoly(T const  c[2]) : coeff{c[0],c[1]}{}
  T operator()(T x) const { return coeff[0]+ x*coeff[1];}
private:
  T coeff[2];
};

/*
template<typename T>
class EstrinPoly<T,2> {
public:
  T operator()(T x) const { return p1(x) + p2(x)*(x*x);}
  EstrinPoly<T,1> p1;
  EstrinPoly<T,0> p2;
};


template<typename T>
class EstrinPoly<T,3> {
public:
  T operator()(T x) const { return p1(x) +p2(x)*(x*x);}
  EstrinPoly<T,1> p1;
  EstrinPoly<T,1> p2;
};


template<typename T>
class EstrinPoly<T,4> {
public:
  EstrinPoly(){}
  EstrinPoly(T coeff[5]) : p1(coeff), p2(coeff+4){};

  T operator()(T x) const { return p1(x) + p2(x)*(x*x)*(x*x);}
  EstrinPoly<T,3> p1;
  EstrinPoly<T,0> p2;
};


template<typename T>
class EstrinPoly<T,5> {
public:
  T operator()(T x) const { return p1(x) + p2(x)*(x*x)*(x*x);}
  EstrinPoly<T,3> p1;
  EstrinPoly<T,1> p2;
};

     
template<typename T>
class EstrinPoly<T,6> {
public:
  T operator()(T x) const { return p1(x) + p2(x)*(x*x)*(x*x);}
  EstrinPoly<T,3> p1;
  EstrinPoly<T,2> p2;
};

template<typename T>
class EstrinPoly<T,7> {
public:
  T operator()(T x) const { return p1(x) + p2(x)*(x*x)*(x*x);}
  EstrinPoly<T,3> p1;
  EstrinPoly<T,3> p2;
};

template<typename T>
class EstrinPoly<T,8> {
public:
  T operator()(T x) const { T x4=(x*x)*(x*x); return p1(x) + p2(x)*x4*x4;}
  EstrinPoly<T,7> p1;
  EstrinPoly<T,0> p2;
};
*/


#include<iostream>

namespace justcomp {
  typedef float Float;
  // typedef double Float;
  Float coeff[] ={1.f,3.f,-3.5f,0.45f,-1.1f,-0.45f,0.3f,-0.23f,0.95f,0.72f, 0.0912f,0.03f,-0.12f,0.67f, 0.11f, -0.023f, 0.056f, 0.011f,
		  1.f,3.f,-3.5f,0.45f,-1.1f,-0.45f,0.3f,-0.23f,0.95f,0.72f, 0.0912f,0.03f,-0.12f,0.67f, 0.11f, -0.023f, 0.056f, 0.011f};
  constexpr int NN=1024*1024;
  Float a[NN], b[NN];
  template<int DEGREE>
  void horner(HornerPoly<Float,DEGREE> const & p) {
    for (int i=0; i!=NN; ++i)
      b[i] = p(a[i]);
  }
  template<int DEGREE>
  void estrin(EstrinPoly<Float,DEGREE> const & p) {
    for (int i=0; i!=NN; ++i)
      b[i] = p(a[i]);
  }
  template<int DEGREE>
  void pade(PadePoly<Float,DEGREE> const & p) {
    for (int i=0; i!=NN; ++i)
      b[i] = p(a[i]);
  }
}

// performance test
#include <x86intrin.h>
inline volatile unsigned long long rdtsc() {
 return __rdtsc();
}


template<int DEGREE, int WHAT>
struct Measure{
  inline void operator()(unsigned long long & t) const;
};

template<int DEGREE>
struct Measure<DEGREE,0> {
  typedef justcomp::Float Float;
  Measure() : p(justcomp::coeff){}
  HornerPoly<Float,DEGREE> p;
  inline
    void operator()(unsigned long long & t) const{
    t -= rdtsc();
    justcomp::horner<DEGREE>(p);
    t += rdtsc();
  }
};


template<int DEGREE>
struct Measure<DEGREE,1> {
  typedef justcomp::Float Float;
  Measure() : p(justcomp::coeff){}
  EstrinPoly<Float,DEGREE> p;
  inline
  void operator()(unsigned long long & t) const{
    t -= rdtsc();
    justcomp::estrin<DEGREE>(p);
    t += rdtsc();
  }
};

template<int DEGREE>
struct Measure<DEGREE,2> {
  typedef justcomp::Float Float;
  Measure() : p(justcomp::coeff){}
  PadePoly<Float,DEGREE> p;
  inline
  void operator()(unsigned long long & t) const{
    t -= rdtsc();
    justcomp::pade<DEGREE>(p);
    t += rdtsc();
  }
};


double  show() {
unsigned long long t=0;
  Measure<5,1> measure;
  measure(t);
  return t;
}

template<int DEGREE, int WHAT>
void perf() {
  Measure<DEGREE,WHAT> measure;
  unsigned long long t=0;
  union { float f;unsigned int i;} x;
  float sum=0;
  long long ntot=0;
  x.f=1.0; // should be 0 but 
  while (x.f<32) { // this is 5*2^23 tests
    ++ntot;
    int i=0;
    while(i<justcomp::NN) { 
      x.i++;
      justcomp::a[i++]=x.f;
      justcomp::a[i++]= (WHAT<2) ? -x.f : 1.f/x.f;
    }
    measure(t);
    for (int i=0; i!=justcomp::NN; ++i)
      sum += justcomp::b[i];
  }
  const char * what[]={"Horner","Estrin","Pade"};
  std::cout << "time for " << what[WHAT] << " degree " << DEGREE << " is "<< double(t)/double(justcomp::NN*ntot) << std::endl;
  std::cout << "sum= " << sum  << std::endl;;
  
}


#include<cassert>
int main() {
  for (int i=0; i!=32; ++i) {
    int k = estrinDetails::ilog2(i);
    std::cout << i<< ":"<< k <<"," << ((i>>(k))<<(k)) <<" ";
  }
  std::cout << std::endl;

  HornerPoly<float,1> h1({-1.f,1.f});
  EstrinPoly<float,1> p1({-1.f,1.f});

  HornerPoly<float,3> h3({-1.f,1.f,0.5f,-0.5f});
  EstrinPoly<float,3> p3({-1.f,1.f,0.5f,-0.5f});


  assert(h1(1.f)==p1(1.f));
  assert(h1(2.f)==p1(2.f));
  assert(h1(-2.f)==p1(-2.f));
  assert(h1(1.f)==0);
  assert(h3(0.f)==-1.f);
  assert(p3(0.f)==-1.f);
  assert(h3(1.f)==0);
  assert(p3(1.f)==0);

  perf<0,0>();
  perf<0,1>();
  perf<0,2>();

  perf<1,0>();
  perf<1,1>();
  perf<1,2>();

  perf<3,0>();
  perf<3,1>();
  perf<3,2>();

  perf<4,0>();
  perf<4,1>();
  perf<4,2>();

  perf<5,0>();
  perf<5,1>();
  perf<5,2>();

  perf<7,0>();
  perf<7,1>();
  perf<7,2>();

  perf<11,0>();
  perf<11,1>();
  perf<11,2>();

  perf<14,0>();
  perf<14,1>();
  perf<14,2>();

  perf<18,0>();
  perf<18,1>();
  perf<18,2>();

  return 0;
} 

     
