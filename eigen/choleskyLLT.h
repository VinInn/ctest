#include <cmath>
template<typename M>
constexpr void choleskyLLT(M& r,int n) {
  using T = typename M::Scalar;
  T d[n];
  for (int j=0; j<n; ++j) {
     r(j,j) = std::sqrt(r(j,j));
     d[j] = T(1)/r(j,j);
     for (int l=j+1; l<n; ++l) {
        r(l,j) *=  d[j];
        T s1 =  0;
        for (int i=0; i<=j; ++i) {
           s1 += r(l,i)*r(j+1,i);
        }
        r(l,j+1) -= s1;
     }
   }
}


