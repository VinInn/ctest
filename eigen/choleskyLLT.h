#include <cmath>

template<typename M>
class  choleskyLLT {
public:
  using T = typename M::Scalar;
  static const int DIM = M::MaxRowsAtCompileTime;

  constexpr choleskyLLT()= default;

  template<typename MM>
  constexpr explicit choleskyLLT(MM const & im) : m(im){}

  constexpr M & matrix() { return m;}
  constexpr M const & matirx() const { return m;}

  constexpr int currSize() const { return m_currSize;}

private:
  M m;
  T d[DIM];
  int m_currSize=0;

}; 


constexpr void choleskyLLT(M& r,int n) {
  using T = typename M::Scalar;
  T d[n];
  for (int j=0; j<n; ++j) {
     d[j] = T(1)/std::sqrt(r(j,j));
     r(j,j)*=d[j];
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


#include <cmath>
template<typename M>
constexpr void choleskyLDLT(M& r,int n) {
  using T = typename M::Scalar;
  for (int j=0; j<n; ++j) {
     r(j,j) = T(1)/r(j,j);
     for (int l=j+1; l<n; ++l) {
        r(l,j) *=  r(j,j);
        T s1 =  0;
        for (int i=0; i<=j; ++i) {
           s1 += r(l,i)*r(j+1,i);
        }
        r(l,j+1) -= s1;
     }
   }
}

