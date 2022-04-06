#include "cstdio"
#include <cmath>
#include "cadna.h"
#include <Eigen/LU>
#include <iostream>


using namespace std;

template<typename FLOAT>
void doit()
{

  cadna_init(-1);


  using Matrix22 = Eigen:: Matrix<FLOAT,2,2>;
  using Matrix23 = Eigen:: Matrix<FLOAT,2,3>;
  using Matrix32 = Eigen:: Matrix<FLOAT,3,2>;

  Matrix23  m;
  Matrix22 y;

  m <<
    0.68,  0.566,  0.823,
   -0.211,  0.597, -0.605;
  y <<-0.33, -0.444,
    0.536,  0.108;


  cout << "Here is the matrix m:" << endl << m << endl;
  cout << "Here is the matrix y:" << endl << y << endl;
  Matrix32 x = m.fullPivLu().solve(y);
  if((m*x).isApprox(y))
  {
    cout << "Here is a solution x to the equation mx=y:" << endl << x << endl;
  }
  else
    cout << "The equation mx=y does not have any solution." << endl;


  cadna_end();
}


int main() {

  doit<float>();
  std::cout << "\nCadna" << std::endl;
  doit<float_st>();

  std::cout <<"\n\nDouble"<< std::endl;
  doit<double>();
  std::cout << "\nCadna" << std::endl;
  doit<double_st>();


  return 0;
}
