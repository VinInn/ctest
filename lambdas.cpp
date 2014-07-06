#include <boost/function_types/is_function_pointer.hpp>
// #include <boost/function_types/is_callable_scalar.hpp>
#include <boost/function_types/function_arity.hpp>

#include <boost/mpl/vector.hpp>
#include <boost/mpl/equal.hpp>

#include <boost/function_types/parameter_types.hpp>

#include <boost/type_traits/remove_pointer.hpp>

#include <boost/bind.hpp>
#include <boost/bind/apply.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/type_traits/function_traits.hpp>
#include <typeinfo>
#include <boost/function.hpp>

#include <boost/assign/std/vector.hpp>
using namespace boost::assign;

#include <iostream>
#include <vector>
#include <valarray>
#include <algorithm>
#include <functional>
#include <iterator>
#include <string>
#include <cmath>

template<typename FP>
void arg2(FP f) {
  // typedef typename boost::remove_pointer<FP>::type F;
  //  boost::function<F> fu(f);
  std::cout << "arity " << boost::function_types::function_arity<FP>::value << std::endl;
  //  std::cout << "arity " << fu.arity << std::endl;
  //  std::cout << typeid(typename boost::detail::function_traits_helper<F>::arg2_type).name() << std::endl;
  // std::cout << typeid(typename boost::function_traits<F>::arg2_type).name() << std::endl;
}

template<typename F>
int arity(F f) {
  return boost::function_types::function_arity<F>::value;
}

template<typename F>
void eval(F f) {
  // std::cout << "eval arity " << boost::function_types::function_arity<typename F::operator() >::value  << std::endl;
  //  std::cout << "eval arity " << arity(&F::operator()) << std::endl;
  double x=2.,y=4;
  std::cout << f(x,y)  << std::endl;
}

void a(int,double){}

struct A {

  void a(int,double){}
  double operator()(int,double) const { return 99.9;}

};

A operator+(A const& x,A const&) { return x;}

A operator+(A const& x, int) { return x;}

double plus(double x, double y) { return x+y;}

int self(int i) { return i;} 


class C; 
typedef C func();
typedef C (*func_ptr)(int);

#include <boost/mpl/assert.hpp>


int main() {

  std::cout << ::boost::function_types::function_arity<func>::value << std::endl;
  // BOOST_MPL_ASSERT_RELATION( ::boost::function_types::function_arity<func_ptr>::value, ==, 1 );
  // BOOST_MPL_ASSERT_RELATION( ::boost::function_types::function_arity<func_ref>::value, ==, 2 );

  std::cout << "is fptr " << ::boost::function_types::is_function_pointer<func>::value   << std::endl;
  std::cout << "is fptr " << ::boost::function_types::is_function_pointer<func_ptr>::value   << std::endl;
  std::cout << "is callable " << ::boost::function_types::is_callable_builtin<func_ptr>::value   << std::endl;
//  std::cout << "is callable " << ::boost::function_types::is_callable_scalar<func_ptr>::value   << std::endl;
  typedef ::boost::function_types::parameter_types<func_ptr>::type V;

  boost::lambda::placeholder1_type X;
  boost::lambda::placeholder2_type Y;
  boost::lambda::placeholder3_type Z;

  int i = 1; 
  (X += 2)(i);         // i is now 3
  (++X, std::cout << X)(i); // i is now 4, outputs 4
  std::cout << std::endl;

  double x=1., y=2.;
  std::cout << (X+Y)(x,y) << std::endl;
  // std::cout << boost::bind<double>((X*X+Y*Y),_1,_2)(x,y) << std::endl;
  std::cout << boost::bind<double>(static_cast<double(*)(double)>(&std::sin),_1)(x) << std::endl;

  A a1;
  eval(a1);

  eval(boost::bind(std::plus<double>(),boost::bind<double>(static_cast<double(*)(double)>(&std::sin),_1),_2));
  //eval(boost::bind<double>(static_cast<double(*)(double)>(&std::sin),_1,_2));

  arg2(a);

  arg2(static_cast<A(*)(A const&,A const&)>(operator+));
  // arg2(boost::bind<double>(static_cast<double(*)(double)>(&std::sin),_1));
  //  arg2(boost::bind<double>((X+Y),_1,_2));
  arg2(&A::a);
  arg2(&A::operator());

  int q=0;
  boost::bind<int>((++X),_1);

  typedef boost::function<int(void)> Fu;
  std::vector<Fu> vf;
  // vf += Fu(boost::bind(self,1)),  Fu(boost::bind(self,2)),  Fu(boost::bind(self,3));  
  vf += boost::bind(self,1), boost::bind(self,2), boost::bind(self,3);  
  std::vector<int> va(vf.size(),0);
  std::vector<int> vb(vf.size(),1);

  std::transform(vf.begin(),vf.end(),va.begin(),boost::bind(boost::apply<int>(),_1));
  // std::for_each(vf.begin(),vf.end(),boost::bind(boost::apply<int>(),_1));
  std::copy(va.begin(),va.end(),std::ostream_iterator<int>(std::cout," "));
  std::cout << std::endl;
  std::transform(vf.begin(),vf.end(),
		 vb.begin(),
		 va.begin(),boost::bind(std::minus<int>(),boost::bind(boost::apply<int>(),_1),_2));
  // std::for_each(vf.begin(),vf.end(),boost::bind(boost::apply<int>(),_1));
  std::copy(va.begin(),va.end(),std::ostream_iterator<int>(std::cout," "));
  std::cout << std::endl;

  return 0;

}
