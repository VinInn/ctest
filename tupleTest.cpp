#include<tuple>
#include<iostream>
#include<functional>
#include<algorithm>

// I prefer 0 to be nop (to avoid code duplication)

// run time iteration
template<class TupleType, size_t N>
struct do_iterate 
{
  template<typename F>
  static void call(TupleType& t, F f) 
  {
    f(std::get<N-1>(t)); 
    do_iterate<TupleType, N-1>::call(t,f); 
  }
 template<typename F>
  static void call(TupleType const & t, F f) 
  {
    f(std::get<N-1>(t)); 
    do_iterate<TupleType, N-1>::call(t,f); 
  }

}; 

template<class TupleType>
struct do_iterate<TupleType, 0> 
{
  template<typename F>
  static void call(TupleType&, F) 
  {}
  template<typename F>
  static void call(TupleType const &, F) 
  {}
}; 

template<class TupleType, typename F>
void iterate_tuple(TupleType& t, F f)
{
  do_iterate<TupleType, std::tuple_size<TupleType>::value>::call(t,f);
}

template<class TupleType, typename F>
void iterate_tuple(TupleType const& t, F f)
{
  do_iterate<TupleType, std::tuple_size<TupleType>::value>::call(t,f);
}
 



template<class TupleType, size_t N=std::tuple_size<TupleType>::value>
struct totDim {
  typedef typename std::tuple_element<N-1,TupleType>::type Elem;
  enum { dim = Elem::dim + totDim<TupleType,N-1>::dim};
};

template<class TupleType>
struct totDim<TupleType, 0>  {
  enum { dim=0};
};

struct A {
  virtual int dimensions() const =0;

  // no need to be virtual: still I wish to test...
  virtual void sum2(int & i) const=0;

};

template<int Dim>
struct B : public A{
  enum { dim=Dim};

  int dimensions() const { return Dim;}
  virtual void sum2(int & i) const { i+=dimensions();}

};

struct tot {
  tot() : t(0){}

  void operator()(A const & a) { t+=a.dimensions();}
  int t;
};

struct tot2 {
  tot2() : t(0){}

  template<int DIM>
  void operator()(B<DIM> const &) { t+=DIM;}
  int t;
};

template<typename T>
void sum2(T& x, T y) { x+=y;}



template< class TupleType>
struct Combi : public A {


  virtual int dimensions() const {
    int tot = 0;
    iterate_tuple(tuple,std::bind(::sum2<int>,std::ref(tot),std::bind(&A::dimensions,std::placeholders::_1)));
    return tot;
  }
  virtual void sum2(int & i) const { i+=dimensions();}


  TupleType tuple;

};


int main() {

  auto tt = std::make_tuple(B<2>(),B<4>(),B<2>(),B<5>());

  tot t;
  iterate_tuple(tt,std::ref(t));  
  tot2 t2;
  iterate_tuple(tt,std::ref(t2));
  int tot3 = 0;
  iterate_tuple(tt,std::bind(&A::sum2,std::placeholders::_1,std::ref(tot3)));
  int tot4 = 0;
  iterate_tuple(tt,std::bind(sum2<int>,std::ref(tot4),std::bind(&A::dimensions,std::placeholders::_1)));
  

  Combi<std::tuple<B<2>,B<4>,B<2>,B<5>>> combi;

  std::cout << "compile time total dimension is " << totDim<std::tuple<B<2>,B<4>,B<2>,B<5>>>::dim << std::endl;
  std::cout << "run time total dimension is " << t.t << std::endl;
  std::cout << "run time total dimension is " << t2.t << std::endl;
  std::cout << "run time total dimension is " << tot3 << std::endl;
  std::cout << "run time total dimension is " << tot4 << std::endl;
  std::cout << "run time total dimension is " << combi.dimensions() << std::endl;

  return 0;

}
