#include <boost/any.hpp>
#include <boost/bind.hpp>
// #include <boost/lambda/lambda.hpp>
#include <boost/function.hpp>
// #include <boost/shared_ptr.hpp>


#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

namespace oldStyle {
  struct FCN {
    virtual ~FCN(){}
    virtual double operator()(double * x) =0;
  };

  struct Evaluator {
    Evaluator() : fcn(0) {}
    void setFCN(FCN * f) {fcn = f;}
    void setPar(std::vector<double> const & v) { par=v;}
    double eval() { return (*fcn)(&par.front());}

    FCN * fcn;
    std::vector<double> par;
  };

}

namespace noVirtual {
 
  struct Evaluator {
    Evaluator() : fcn(0) {}
    template<typename FCN>
    void setFCN(FCN f) {fcn = f;}
    void setPar(std::vector<double> const & v) { par=v;}
    double eval() { return fcn(&par.front());}

    boost::function<double(double *)> fcn;
    std::vector<double> par;
  };

}


inline double quad(double * x) {
  return x[0]*x[0]+x[1]*x[1];
}

struct Quad : public  oldStyle::FCN {
  double operator()(double * x) {
    return x[0]*x[0]+x[1]*x[1];
  }
};

  
int main() {

  double z;
  std::vector<double> x(2);x[0]=2;x[1]=3;

  Quad q;
  oldStyle::Evaluator oE;
  oE.setPar(x);
  oE.setFCN(&q);
  z=oE.eval();
  std::cout << z << std::endl;

  noVirtual::Evaluator nE;
  nE.setPar(x);
  nE.setFCN(quad);
  z=nE.eval();
  std::cout << z << std::endl;

  return 0;

}
