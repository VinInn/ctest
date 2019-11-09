#include<cmath>

template<typename T>
constexpr void rotg(T& sa, T& sb, T& c, T& s) {

  // construct givens plane rotation
  // adapted from blas (simplified, assume no zero in input)

  auto roe = (std::abs(sa) > std::abs(sb)) ? sa : sb;
  auto r = std::copysign(std::sqrt(sa*sa + sb*sb),roe);
  auto ir = T(1)/r;
  c = sa*ir;
  s = sb*ir;
  auto z = (std::abs(sa) > std::abs(sb)) ? s : T(1)/c;
  sa = r;
  sb = z;
}

template<typename M>
constexpr void choleskyShiftDown(M& r,int k, int l) {
  
  // adapted from linpack chex
  //the rows are rearranged in the following order:
  // 0,...,k-1,l,k,k+1,...,l-1,l+1,...,p-1.

  assert(l>k);

  using T = typename M::Scalar;
  int p = r.rows();
  
  // we assume r to be lower triangular

  auto lmk = l - k;

  T s[p];
  // reorder the column
  for (int i = 0; i<=l; ++i) {
     s[i] = r(l,l-i);
  }
  for (int jj = k; jj<l; ++jj) {
     auto j = l-1 - jj + k;
     for (int i = 0; i<=j; ++i)
       r(j+1,i) = r(j,i);
     r(j+1,j+1) = T(0);
   }
   for (int i = 0; i<k; ++i) {
      r(k,i) = s[l-i];
   }

   //calculate the rotations.
   T c[p];
   auto t = s[0];
   for (int i = 0; i<lmk; ++i) { 
      rotg(s[i+1],t,c[i],s[i]);
      t = s[i+1];
   }
   r(k,k) = t;
   for (auto j=k+1; j<p; ++j) {
     auto il = std::max(0,l-j);
     for (int ii = il; ii<lmk; ++ii) {
        auto i = l - ii;
        assert(i>0);
        assert(j>=i);
        t =  c[ii]*r(j,i-1) + s[ii]*r(j,i);
        r(j,i) = -c[ii]*r(j,i) + s[ii]*r(j,i-1);
        r(j,i-1) = t;
     }
   }
}



template<typename M>
constexpr void choleskyShiftUp(M& r,int k, int l) {
  // adampted from linpack chex
  //the rows are rearranged in the following order:
  // 0,...,k-1,k+1,k+2,...,l,k,l+1,...,p-1.  

  assert(l>k);

  using T = typename M::Scalar;
  int p = r.rows();

  // we assume r to be lower triangular

  auto kp1 = k + 1;
  auto lmk = l - k;

  T s[p];
  // reorder the columns
  for (int i = 0; i<kp1; ++i) {
     auto ii = lmk + i;
     s[ii] = r(k,i);
  }
  for (int j = k; j<l; ++j) {
    for (int i = 0; i<=j; ++i)
       r(j,i) = r(j+1,i);
    auto jj = j - k;
    s[jj] = r(j+1,j+1);
  }
  for(int i = 0; i<kp1; ++i) {
    auto ii = lmk + i;
    r(l,i) = s[ii];
  }
  for (int i = kp1; i<=l; ++i)
     r(l,i) = T(0);

  //  reduction loop.
  T c[p];
  for (int j = k; j<p; ++j) {
      if (j> k) {
        //apply the rotations.
        // auto iu = std::min(j-1,l-1);
        auto iu = std::min(j,l);
        for (int i=k; i<iu; ++i) {
          auto ii = i - k;
          assert(ii<j-k);
          assert(j>=i+1);
          auto t = c[ii]*r(j,i) + s[ii]*r(j,i+1);
          r(j,i+1) = -c[ii]*r(j,i+1) + s[ii]*r(j,i);
          r(j,i) = t;
        } 
      }
      if (j < l) {
        auto jj = j - k;
        auto t = s[jj];
        rotg(r(j,j),t,c[jj],s[jj]);
      }
  }
}

