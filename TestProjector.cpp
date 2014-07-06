#include "Projector.h"
#include <iostream>

typedef ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > AlgebraicSymMatrix22;
typedef ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepSym<double,5> > AlgebraicSymMatrix55;

typedef ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepStd<double,5,5> > AlgebraicMatrix55;  

typedef ROOT::Math::SMatrix<double,2,5,ROOT::Math::MatRepStd<double,2,5> > MatD5;
typedef ROOT::Math::SMatrix<double,5,2,ROOT::Math::MatRepStd<double,5,2> > Mat5D;
 
#include "RealTime.h"


template<typename Pr> 
void go(Pr const & P, AlgebraicSymMatrix55 const & C);

int main() {

  double aH1[] =
    {0, 0, 0, 1, 0, 
     0, 0, 0, 0, 1};
  
  MatD5 H1(aH1,10);
 
  double aH2[] =
    {0, 0, 1, 0, 0, 
     0, 0, 0, 0, 1};
  
  MatD5 H2(aH2,10);
  

  double aC[] = 
    {0.60194, -0.14262, 0.020447, 2.2361, -0.27993,
     -0.14262, 0.033806, -0.0044397, -0.52993, 0.059251, 
     0.020447, -0.0044397, 0.014513, 0.073261, -0.25078, 
     2.2361, -0.52993, 0.073261, 8.3081, -0.99288, 
     -0.27993, 0.059251, -0.25078, -0.99288, 4.3449
    };

  AlgebraicMatrix55 fC(aC,25);
  AlgebraicSymMatrix55 C = fC.LowerBlock();

  // using ROOT::Math::Similarity;

  {

    perftools::TimeType start = perftools::realTime();
    AlgebraicSymMatrix22 M =  Similarity(H1,C);
    perftools::TimeType end = perftools::realTime();
    std::cout << "Similarity   real time " << 1.e-9*(end-start) << std::endl;
 
    M.Print(std::cout); 
    std::cout << std::endl;
  }

 {

    perftools::TimeType start = perftools::realTime();
    AlgebraicSymMatrix22 M =  Similarity(H2,C);
    perftools::TimeType end = perftools::realTime();
    std::cout << "Similarity   real time " << 1.e-9*(end-start) << std::endl;
 
    M.Print(std::cout); 
    std::cout << std::endl;
  }

 std::cout << std::endl;

  ProjectorBySubMatrix<2> P1(3);

  go(P1,C);

  size_t i[] = {2,4};
  ProjectorByGSlice<2> P2(i);

  go(P2,C);

  ProjectorBySimilarity<2> P3(H2);

  go(P3,C);

  return 0;

}

 
template<typename Pr> 
void go(Pr const & P, AlgebraicSymMatrix55 const & C) {
  
  Projector<2> const & PP =P;
  
  
  {
    
    perftools::TimeType start = perftools::realTime();
    AlgebraicSymMatrix22 M = P.project(C);
    perftools::TimeType end = perftools::realTime();
    std::cout << "Project    real time " << 1.e-9*(end-start) << std::endl;
 
    M.Print(std::cout); 
    std::cout << std::endl;
  }
 
  {

    perftools::TimeType start = perftools::realTime();
    AlgebraicSymMatrix22 M = PP.project(C);
    perftools::TimeType end = perftools::realTime();
    std::cout << "Project  vir real time " << 1.e-9*(end-start) << std::endl;
 
    M.Print(std::cout); 
    std::cout << std::endl;
  }
 
   
  {

    perftools::TimeType start = perftools::realTime();
    AlgebraicSymMatrix22 M = Similarity(PP,C);
    perftools::TimeType end = perftools::realTime();
    std::cout << "Similarity P real time " << 1.e-9*(end-start) << std::endl;
 
    M.Print(std::cout); 
    std::cout << std::endl;
  }
    std::cout << std::endl;
    std::cout << std::endl;

}
