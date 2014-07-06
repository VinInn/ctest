#include <boost/any.hpp>
#include <boost/bind.hpp>
// #include <boost/lambda/lambda.hpp>
#include <boost/function.hpp>
// #include <boost/shared_ptr.hpp>


#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

template<typename T>
std::ostream & print(std::ostream& co, T const & t, char const * sep="") {
  return co << t << sep;
}

struct A {
  A(){}
  A(int i,int j) {
    k=i+j;
  }
  static A fill(int i,int j) { return A(i,j);}
  int k;
};


int main() {

  int v[2] ={1,2};

  A a[2];
  std::string s("hi");
  boost::bind(print<std::string>,boost::ref(std::cout),_1," ")(s);

 std::for_each(v,v+2,  
	       boost::bind(print<int>,boost::ref(std::cout),_1," "));
 std::cout << std::endl;

 std::transform(v,v+2,a,
		boost::bind(A::fill,_1,_1)
		);

 std::find_if(a,a+2, (boost::bind(&A::k,_1)>1) && (boost::bind(&A::k,_1)<3) );

 // std::for_each(v,v+2, std::cout << boost::lambda::_1 << std::endl);
  // std::for_each(v,v+2, boost::lambda::_1 =1);

  //   (std::cout <<  boost::lambda::_1 << std::endl)("hello");


  return 0;

}
