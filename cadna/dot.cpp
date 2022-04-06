#include <iostream>
#include <Eigen/Dense>
#include <cadna.h>
using namespace Eigen ;
using namespace std;

void doNative() {

  cout << "Native" << std::endl;
  Matrix < float, 3, 1> v(1 ,2 ,3);
  Matrix < float, 3, 1> w(0 ,1 ,2);
  cout << "Dot product : " << v.dot (w) << endl ;
}

void doCadna() {

  cout << "\nCadna" << endl;
  cadna_init ( -1);
  Matrix < float_st , 3, 1> v(1 ,2 ,3);
  Matrix < float_st , 3, 1> w(0 ,1 ,2);
  cout << "Dot product : " << v.dot (w) << endl ;

  cadna_end();

}

int main () {
  doNative();
  doCadna();

  return 0;
}
