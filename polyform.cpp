#include <iostream>
#include <boost/fusion/sequence.hpp>
#include <cmath>
#include <algorithm>
#include <functional>
#include <iterator>

#include <boost/bind.hpp>
#include <boost/function.hpp>



#include <boost/assign/std/vector.hpp>
#include <vector>

template<int N> struct Seq;

template<> struct Seq<0> {};

template<int N> struct Seq {
  Seq() {}
  Seq(double d) { data[0]=d;}
  Seq(double d[]) { std::copy(d,d+N,data);}

  template<int K, int L> 
  Seq(Seq<K> const & k, Seq<L> const & l) {
    std::copy(k.data,k.data+K,data);
    std::copy(l.data,l.data+L,data+K);
  }
  template<int K> 
  Seq(Seq<K> const & k, Seq<0>) {
    std::copy(k.data,k.data+K,data);
  }
  
  Seq(Seq<N-1> const & k, double d) {
    std::copy(k.data,k.data+(N-1),data);
    data[N-1]=d;
  }
  Seq(double d, Seq<N-1> const & k) {
    std::copy(k.data,k.data+(N-1),data+1);
    data[0]=d;
  }

  Seq(double d, Seq<0>) {
    data[0]=d;
  }

  double operator[](int i) const { return data[i];}
  double data[N];
};



template<> struct Seq<1> {
  Seq() {}
  Seq(double d) { data[0]=d;}
  Seq(double d[]) { data[0]= d[0];}
  
  
  Seq(double d, Seq<0>) {
    data[0]=d;
  }

  Seq(Seq<0>, double d) {
    data[0]=d;
  }

  double operator[](int i) const { return data[i];}
  double data[1];
};


template<int N, int M>
Seq<N+M> operator,(Seq<N> const & n, Seq<M> const & m) {
  return Seq<N+M>(n,m); 
} 
template<int N>
Seq<N+1> operator,(Seq<N> const & n, double d) {
  return Seq<N+1>(n,d); 
} 
template<int N>
Seq<N+1> operator,(double d, Seq<N> const & n) {
  return Seq<N+1>(d,n); 
} 
template<int N>
Seq<N+1> operator,(Seq<N> const & n, Seq<1> d) {
  return Seq<N+1>(n,d.data[0]); 
} 
template<int N>
Seq<N+1> operator,(Seq<1> d, Seq<N> const & n) {
  return Seq<N+1>(d.data[0],n); 
} 


template<int N> 
class Polynomial {
public:
  double operator()(double a0, double x)  {return a0; }
  double operator()(double a0, double a1, double x) {
    return a0 + p(a1,x)*x;
  }
  double operator()(double a0, double a1, double a2, double x) {
    return a0 + p(a1,a2,x)*x;
  }
  double operator()(double a0, double a1, double a2, double a3, double x) {
    return a0 + p(a1,a2,a3,x)*x;
  }

  double operator()(double a0, double a1, double a2, double a3, double a4, double x) {
    return a0 + p(a1,a2,a3,a4,x)*x;
  }

  template<typename Iter>
  double eval(Iter a, double x) { return *a + p.eval(++a, x)*x; }

  Polynomial<N-1> p;
};

template<> class Polynomial<0> {
public:
  double operator()(double a0, double x)  {return a0; }
  template<typename Iter>
    double eval(Iter a, double)  {return *a; }

};


/*
class Polynomial<2> {

  double operator(double a0, double a1, double a2, double x)  {return a0 + (a1+a2*x)*x};

};
*/


template<int N>
void print(Seq<N> const & s) {
  std::copy(s.data,s.data+N,std::ostream_iterator<double>(std::cout," "));
  std::cout << std::endl;
}

double f(double a, double x) {
  return sin(a*x);
}

double g(double x) {
  return std::exp(-(x*x));
}

int main() {
  double a[3]= {0.5,-1.,0};

  Seq<1> s1(a);
  Seq<2> s2(a+1);
  Seq<3> s3(s1,s2);
  Seq<3> s31((s1,s2));
  Seq<4> s4((s3,4.5));
  print (s4);
  print ((s3,4.5));
  print (( Seq<0>(), 0.5,-1,0,4.5));

  // double a[3]= {0.5,-1.,0};
  Polynomial<0> p0;
  Polynomial<1> p1;
  Polynomial<2> p2;
  std::cout << p0.eval(a,2) << std::endl;
  std::cout << p1.eval(a,2) << std::endl;
  std::cout << p2.eval(a,2) << std::endl;
  std::cout << p0(0.5,2) << std::endl;
  std::cout << p1(0.5,-1.0,2) << std::endl;
  std::cout << p2(0.5,-1.0,0.,2) << std::endl;

  std::cout << boost::bind(g,boost::bind(&Polynomial<1>::eval<double *>,Polynomial<1>(),_1,_2))(a,2.) << std::endl;

  boost::function<double(double*,double)> g1(boost::bind(g,boost::bind(&Polynomial<1>::eval<double *>,Polynomial<1>(),_1,_2)));

  double a[3]= {0.5,-1.,-0.5};
  boost::function<double(double)> b2(boost::bind(g1,a,_1));

  std::cout << b2(-0.5) << " " << b2(0.5) << " " << b2(1.5) << std::endl; 
  a[0]=-0.5;
  std::cout << b2(-0.5) << " " << b2(0.5) << " " << b2(1.5) << std::endl; 

  double par[6]= {0.5,-1.,-0.5,-1.,1.,0.2};
  // boost::function<double(double)> b2(&Polynomial<1>::eval<double *>,Polynomial<1>(),boost::bind(g1,par,_1),(boost::bind(g1,par+2,_1));


  // boost::fusion


  return 0;

}
