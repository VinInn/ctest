/*
define (binomial n k)
;; Helper function to compute C(n,k) via forward recursion
  (define (binomial-iter n k i prev)
    (if (>= i k)
      prev
     (binomial-iter n k (+ i 1) (/ (* (- n i) prev) (+ i 1)))))
;; Use symmetry property C(n,k)=C(n, n-k)
  (if (< k (-  n k))
    (binomial-iter n k 0 1)
    (binomial-iter n (- n k) 0 1)))
*/



unsigned choose(unsigned n, unsigned k) {
  unsigned r = 1;
  unsigned d;
  if (k > n) return 0;
  for (d=1; d <= k; d++) {
    r *= n--;
    r /= d;
  }
  return r;
}


constexpr long long binomialIter(long long n, long long k, long long i, long long prev) {
  return (i>=k) ? prev : binomialIter(n,k, i+1, ((n-i)*prev)/(i+1));
}

constexpr long long binomial(long long n, long long k) {
  return (k>n) ? 0 : ( (k<(n-k)) ? binomialIter(n,k,0,1) :  binomialIter(n,(n-k),0,1));
}

#include <iostream>
int main() {

  for (int k=0; k!=10; ++k) {
    for (int n=0; n!=10; ++n) 
      std::cout << choose(n,k) << " " << binomial(n,k) <<", ";
    std::cout << std::endl;
  }

  return 0;
}


double bha(double x, double const * c, int degree) {
  //  double f=1.;
  //for (int k=1; k!=degree; ++k)
  // f*=(1.-x);

 double ret=c[0];
 for (int k=0; k!=degree; ++k) {
   ret += c[k+1]*binomial(degree,k+1)*x;
   x*=x;
 }
 return ret;
}

double foo(double x, double const * c) {
  return bha(x,c,5);
}

double bho(double x, double const * c) {
  int degree=5;
  double f=1.;
  for (int k=1; k!=degree; ++k)
    f*=(1.-x);
  x = x/(1-x);
  double ret=c[degree];
  for (int k=degree; k!=0; --k) {
    ret += ret*x+c[k-1];
  }
  return f*ret;
}
