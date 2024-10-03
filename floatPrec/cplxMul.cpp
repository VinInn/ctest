#include <cstdio>
#include <complex>


/*
int main(int argc, char** argv)
{
 std::complex<float> z1{0x1.6912cap-41f,-0x1.69020ap-41f};
 std::complex<float> z2{0x1.d16ed4p-85f,0x1.6b497ep-110f};
  if (argc>1) z1 +=  atof(argv[1]);
  if (argc>2) z2 += atof(argv[2]);
  auto z3 = z1;
  auto z4 = z2;
  if (argc>3) z3 +=  atof(argv[3]);
  if (argc>4) z4 += atof(argv[4]);
 std::complex<float> t1 = z1 * z2;
 std::complex<float> t2 = z4 * z3;
 printf ("t1=(%a,%a)\n", t1.real(), t1.imag());
 printf ("t2=(%a,%a)\n", t2.real(), t2.imag());

 return 0;
}
*/


// ./a.out 0x1.6912cap-41f -0x1.69020ap-41f 0x1.d16ed4p-85f 0x1.6b497ep-110f
// ./a.out 0x1.d16ed4p-85f 0x1.6b497ep-110f 0x1.6912cap-41f -0x1.69020ap-41f 
int main(int argc, char** argv)
{
 std::complex<float> z1{atof(argv[1]),atof(argv[2])};
 std::complex<float> z2{atof(argv[3]),atof(argv[4])};
 std::complex<float> t1 = z1 * z2;
 printf ("t1=(%a,%a)\n", t1.real(), t1.imag());
 return 0;
}

