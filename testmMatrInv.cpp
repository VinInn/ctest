#include <Math/SVector.h>
#include <Math/SMatrix.h>
#include "choleskyInverse.h"

#include<iostream>
#include "RealTime.h"

/// ============= When we need templated root objects 
template <unsigned int D1, unsigned int D2=D1> struct Algebraic {
  //  typedef typename ROOT::Math::SVector<double,D1> Vector;
  typedef typename ROOT::Math::SMatrix<double,D1,D1,ROOT::Math::MatRepSym<double,D1> > SymMatrix;
  typedef typename ROOT::Math::SMatrix<double,D1,D2,ROOT::Math::MatRepStd<double,D1,D2> > Matrix;
};


template<size_t D>
void invertit(ROOT::Math::SMatrix<double,D,D,ROOT::Math::MatRepSym<double,D> >& m) {

  typedef typename ROOT::Math::SMatrix<double,D,D,ROOT::Math::MatRepSym<double,D> > SymMatrix;


  std::cout << m << std::endl;
 { 
   std::cout << "Invert" << std::endl;
   SymMatrix a=m;
   a.Invert();
   std::cout << a << std::endl;
   a.Invert();
   std::cout << a << std::endl;

   perftools::TimeType start = perftools::realTime();
   for (int i=0; i<10000;++i)    a.Invert();
   perftools::TimeType end = perftools::realTime();
   std::cout << D << "  Invert real time " << 1.e-9*(end-start) << std::endl;
   std::cout << a << std::endl;
 }

/* 
 { 
   std::cout << "Sinvert" << std::endl;
   SymMatrix a=m;
   a.Sinvert();
   std::cout << a << std::endl;
   a.Sinvert();
   std::cout << a << std::endl;
   perftools::TimeType start = perftools::realTime();
   for (int i=0; i<10000;++i)    a.Sinvert();
   perftools::TimeType end = perftools::realTime();
   std::cout << D << " Sinvert real time " << 1.e-9*(end-start) << std::endl;
   std::cout << a << std::endl;
  
 }
*/
 { 
   std::cout << "VIchol" << std::endl;
   SymMatrix a=m;
   choleskyInverse<SymMatrix,D>(a);
   std::cout << a << std::endl;
   choleskyInverse<SymMatrix,D>(a);
   std::cout << a << std::endl;
   perftools::TimeType start = perftools::realTime();
   for (int i=0; i<10000;++i)    choleskyInverse<SymMatrix,D>(a);
   perftools::TimeType end = perftools::realTime();
   std::cout << D << " VIchol  real time " << 1.e-9*(end-start) << std::endl;
   std::cout << a << std::endl;
 
 }
 
 
}

template<size_t D>
void invert() {
 typedef typename ROOT::Math::SMatrix<double,D,D,ROOT::Math::MatRepSym<double,D> > SymMatrix;

 SymMatrix m;
 double one = 1.e-6;
 for (size_t i=0; i<D;++i)
   m(i,i) = one = 1./one;

 invertit<D>(m);

}


int main() {

  /*
  for (size_t i=0; i<5; ++i) 
    for (size_t j=i+1; j<5; ++j) 
      std::cout << i <<","<<j << std::endl;
  */


  invert<3>();
  invert<6>();
  invert<8>();

  typedef Algebraic<5,5>::Matrix Mat55;
  typedef Algebraic<5,5>::SymMatrix SMat55;
  typedef Algebraic<2,2>::SymMatrix SMat22;

  double aV[] =
    {1.7827e-05, 0.00016332, 0.0026591};

  SMat22 V(aV,3); 

  double aC[] = 
    {0.60194, -0.14262, 0.020447, 2.2361, -0.27993,
     -0.14262, 0.033806, -0.0044397, -0.52993, 0.059251, 
      0.020447, -0.0044397, 0.014513, 0.073261, -0.25078, 
     2.2361, -0.52993, 0.073261, 8.3081, -0.99288, 
     -0.27993, 0.059251, -0.25078, -0.99288, 4.3449};

  Mat55 fC(aC,25);
  std::cout << "full matrix " << fC << std::endl;
  SMat55 C1 = fC.LowerBlock();
  std::cout <<"=LB\n" << C1 << std::endl;
  SMat55 C2 = fC.UpperBlock();
  std::cout << "=UB\n" << C2 << std::endl;
  SMat55 C3(fC.LowerBlock(),true);
  std::cout <<"LB constr\n" << C3 << std::endl;
  SMat55 C4(fC.UpperBlock(),false);
  std::cout << "UB constr \n" << C4 << std::endl;

  SMat55 C(fC.LowerBlock(),true);


  invertit<2>(V);
  invertit<5>(C);


}
