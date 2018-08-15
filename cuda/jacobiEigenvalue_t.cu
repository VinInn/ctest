#include "jacobi_eigenvalue.h"

#include "cuda/api_wrappers.h"

template<int N>
__global__
void jacobiEigenvalue (double * a, int it_max, double * v, 
                       double * d, int * it_num, int * rot_num ) {

 jacobi_eigenvalue<N>(N, a, it_max, v,d, *it_num, *rot_num);

}



# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>
# include <cstring>

using namespace std;


int main ( );
void test01 ( );
void test02 ( );
void test03 ( );

//****************************************************************************80

int main ( )

//****************************************************************************80
//
//  Purpose:
//
//    MAIN is the main program for JACOBI_EIGENVALUE_PRB.
//
//  Discussion:
//
//    JACOBI_EIGENVALUE_PRB tests the JACOBI_EIGENVALUE library.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    15 July 2013
//
//  Author:
//
//    John Burkardt
//
{
  timestamp ( );
  cout << "\n";
  cout << "JACOBI_EIGENVALUE_PRB\n";
  cout << "  C++ version\n";
  cout << "  Test the JACOBI_EIGENVALUE library.\n";

  test01 ( );
/*
  test02 ( );
  test03 ( );
*/
//
//  Terminate.
//
  cout << "\n";
  cout << "JACOBI_EIGENVALUE_PRB\n";
  cout << "  Normal end of execution.\n";
  cout << "\n";
  timestamp ( );

  return 0;
}
//****************************************************************************80

void test01 ( )

//****************************************************************************80
//
//  Purpose:
//
//    TEST01 uses a 4x4 test matrix.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    15 July 2013
//
//  Author:
//
//    John Burkardt
//
{

  if (cuda::device::count() == 0) {
	std::cerr << "No CUDA devices on this system" << "\n";
	exit(EXIT_FAILURE);
  }

  auto current_device = cuda::device::current::get(); 




# define N 4

  double a[N*N] = {
      4.0,  -30.0,    60.0,   -35.0, 
    -30.0,  300.0,  -675.0,   420.0, 
     60.0, -675.0,  1620.0, -1050.0, 
    -35.0,  420.0, -1050.0,   700.0 };
  double d[N];
  double error_frobenius;
  int it_max;
  int it_num;
  int n = N;
  int rot_num;
  double v[N*N];

  cout << "\n";
  cout << "TEST01\n";
  cout << "  For a symmetric matrix A,\n";
  cout << "  JACOBI_EIGENVALUE computes the eigenvalues D\n";
  cout << "  and eigenvectors V so that A * V = D * V.\n";

  r8mat_print ( n, n, a, "  Input matrix A:" );

  it_max = 100;

  auto d_d = cuda::memory::device::make_unique<double[]>(current_device, N);
  auto v_d = cuda::memory::device::make_unique<double[]>(current_device, N*N);
  auto a_d = cuda::memory::device::make_unique<double[]>(current_device, N*N);

  auto rot_d = cuda::memory::device::make_unique<int[]>(current_device, 1);
  auto it_d = cuda::memory::device::make_unique<int[]>(current_device, 1);

  cuda::memory::copy(a_d.get(), a, N*N*sizeof(double));

   int threadsPerBlock =32;
   int blocksPerGrid = 1;
   cuda::launch(
                jacobiEigenvalue<N>,
                { blocksPerGrid, threadsPerBlock },
                a_d.get(), it_max, v_d.get(), d_d.get(), it_d.get(), rot_d.get())
               ;

  cuda::memory::copy(&it_num, it_d.get(), 4);
  cuda::memory::copy(&rot_num, rot_d.get(), 4);
  cuda::memory::copy(v, v_d.get(), N*N*sizeof(double));
  cuda::memory::copy(d, d_d.get(), N*sizeof(double));


  cout << "\n";
  cout << "  Number of iterations = " << it_num << "\n";
  cout << "  Number of rotations  = " << rot_num << "\n";

  r8vec_print ( n, d, "  Eigenvalues D:" );

  r8mat_print ( n, n, v, "  Eigenvector matrix V:" );
//
//  Compute eigentest.
//
  error_frobenius = r8mat_is_eigen_right ( n, n, a, v, d );
  cout << "\n";
  cout << "  Frobenius norm error in eigensystem A*V-D*V = " 
       << error_frobenius << "\n";

  return;
# undef N
}
//****************************************************************************80

void test02 ( )

//****************************************************************************80
//
//  Purpose:
//
//    TEST02 uses a 4x4 test matrix.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    15 July 2013
//
//  Author:
//
//    John Burkardt
//
{
# define N 4

  double a[N*N] = {
    4.0, 0.0, 0.0, 0.0, 
    0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 3.0, 0.0, 
    0.0, 0.0, 0.0, 2.0 };
  double d[N];
  double error_frobenius;
  int it_max;
  int it_num;
  int n = N;
  int rot_num;
  double v[N*N];

  cout << "\n";
  cout << "TEST02\n";
  cout << "  For a symmetric matrix A,\n";
  cout << "  JACOBI_EIGENVALUE computes the eigenvalues D\n";
  cout << "  and eigenvectors V so that A * V = D * V.\n";
  cout << "\n";
  cout << "As a sanity check, input a diagonal matrix.\n";

  r8mat_print ( n, n, a, "  Input matrix A:" );

  it_max = 100;

  jacobi_eigenvalue<N> ( n, a, it_max, v, d, it_num, rot_num );

  cout << "\n";
  cout << "  Number of iterations = " << it_num << "\n";
  cout << "  Number of rotations  = " << rot_num << "\n";

  r8vec_print ( n, d, "  Eigenvalues D:" );

  r8mat_print ( n, n, v, "  Eigenvector matrix V:" );
//
//  Compute eigentest.
//
  error_frobenius = r8mat_is_eigen_right ( n, n, a, v, d );
  cout << "\n";
  cout << "  Frobenius norm error in eigensystem A*V-D*V = " 
       << error_frobenius << "\n";

  return;
# undef N
}
//****************************************************************************80

void test03 ( )

//****************************************************************************80
//
//  Purpose:
//
//    TEST03 uses a 5x5 test matrix.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    15 July 2013
//
//  Author:
//
//    John Burkardt
//
{
# define N 5

  double a[N*N];
  double d[N];
  double error_frobenius;
  int i;
  int it_max;
  int it_num;
  int j;
  int n = N;
  int rot_num;
  double v[N*N];

  cout << "\n";
  cout << "TEST03\n";
  cout << "  For a symmetric matrix A,\n";
  cout << "  JACOBI_EIGENVALUE computes the eigenvalues D\n";
  cout << "  and eigenvectors V so that A * V = D * V.\n";
  cout << "\n";
  cout << "  Use the discretized second derivative matrix.\n";

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < n; i++ )
    {
      if ( i == j )
      {
        a[i+j*n] = -2.0;
      }
      else if ( i == j + 1 || i == j - 1 )
      {
        a[i+j*n] = 1.0;
      }
      else
      {
        a[i+j*n] = 0.0;
      }
    }
  }

  r8mat_print ( n, n, a, "  Input matrix A:" );

  it_max = 100;

  jacobi_eigenvalue<N> ( n, a, it_max, v, d, it_num, rot_num );

  cout << "\n";
  cout << "  Number of iterations = " << it_num << "\n";
  cout << "  Number of rotations  = " << rot_num << "\n";

  r8vec_print ( n, d, "  Eigenvalues D:" );

  r8mat_print ( n, n, v, "  Eigenvector matrix V:" );
//
//  Compute eigentest.
//
  error_frobenius = r8mat_is_eigen_right ( n, n, a, v, d );
  cout << "\n";
  cout << "  Frobenius norm error in eigensystem A*V-D*V = " 
       << error_frobenius << "\n";

  return;
# undef N
}
