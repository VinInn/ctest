#include "cstdio"
#include <cmath>
#include "cadna.h"
#include <Eigen/LU>
#include <iostream>

#define IDIM 4


template<typename FLOAT>
void doit()
{

  cadna_init(-1);


using  Matrix4f = Eigen::Matrix<FLOAT, IDIM, IDIM>;
using  Vector4f = Eigen::Vector<FLOAT, IDIM>;

  Matrix4f a;
  a << 
      21.0, 130.0,       0.0,    2.1,
      13.0,  80.0,   4.74e+8,  752.0,
       0.0,  -0.4, 3.9816e+8,    4.2,
       0.0,   0.0,       1.7, 9.0E-9;


  Matrix4f t = a.transpose();

  Matrix4f at = a*t;

  std::cout << "a*T\n" << at << std::endl; 

  cadna_end();
  cadna_init(-1);

  
  Vector4f b = {153.1,849.74,7.7816, 2.6e-8};

   Vector4f  xsol={1., 1., 1.e-8,1.};

  printf("------------------------------------------------------\n");
  printf("| Solving a linear system using Gaussian elimination |\n");
  printf("| with partial pivoting                              |\n");
  printf("------------------------------------------------------\n");
 
  std::cout << "Here is the invertible matrix A:\n" << a << std::endl;
  std::cout << "Here is the matrix B:\n" << b << std::endl;
  Vector4f x = a.fullPivLu().solve(b);  // a.lu().solve(b);
  std::cout << "Here is the (unique) solution X to the equation AX=B:\n" << x << std::endl;
  std::cout << "Relative error: " << (a*x-b).norm() / b.norm() << std::endl;

  
  for(int i=0;i<IDIM;i++)
    printf("x_sol(%d) = %s (exact solution: xsol(%d)= %s)\n",
	   i,strp(float_st(x[i])),i,strp(float_st(xsol[i])));
 
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
