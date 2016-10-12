#include<cmath>
#include<limits>
#include<cstring>
#include<cstdlib>
#include<cstdio>

#include <cerrno>
#include <csignal>

#include<iostream>

struct sigaction ignaction, termaction, newaction, oldaction;

void termSignalHandler(int sig) {
  std::cerr << "error " << sig << std::endl;
}

double test(int i0, int n, double a)
{
double sum = 0.0;
int i;

for(i=i0; i<n; ++i)
  {
    float x = logf((float)i);
    // sum += std::isnan(x) ? 0 : x;
    sum += (x!=x) ? 0 : x;
  }

return sum;
}

// #include <features.h>
#include <fenv.h>

void testEx() {

int e = std::fetestexcept(FE_ALL_EXCEPT);
    if (e & FE_DIVBYZERO) {
        std::cout << "division by zero\n";
    }
    if (e & FE_INEXACT) {
        std::cout << "inexact\n";
    }
    if (e & FE_INVALID) {
        std::cout << "invalid\n";
    }
    if (e & FE_UNDERFLOW) {
        std::cout << "underflow\n";
    }
    if (e & FE_OVERFLOW) {
        std::cout << "overflow\n";
    }

}


int main(void) {


  feenableexcept( FE_DIVBYZERO );
  feenableexcept( FE_INVALID );
  feenableexcept( FE_OVERFLOW );
  feenableexcept( FE_UNDERFLOW );

  testEx();

  sigset_t *def_set;
  def_set=&termaction.sa_mask;
  sigfillset(def_set);
  sigdelset(def_set,SIGFPE);
  termaction.sa_handler=termSignalHandler;
  termaction.sa_flags=0;
  sigaction(SIGFPE, &termaction,&oldaction);

  std::cout << log(0.) << " " << exp(log(0.)) << ' ' << sqrt(-1.) << std::endl;

   testEx();

  ::printf("test(4, 6, 0) = %f\n", test(4,6,0));
  ::printf("test(0, 2, 0) = %f\n", test(0,2,0));
  ::printf("test(-2, 3, 0) = %f\n", test(-2,3,0));
  ::printf("nan is  = %f\n", test(0, 2, 0) - test(-2,3,0));

  testEx();

  return 0;
}

