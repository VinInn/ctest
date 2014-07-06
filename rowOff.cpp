#include <cstddef>
#include <utility>
#include <type_traits>
#include <array>

namespace rowOffsetsUtils {
  
  ///////////
  // Some meta template stuff
  template<int...> struct indices{};
  
  template<int I, class IndexTuple, int N>
  struct make_indices_impl;
  
  template<int I, int... Indices, int N>
  struct make_indices_impl<I, indices<Indices...>, N>
  {
    typedef typename make_indices_impl<I + 1, indices<Indices..., I>,
				       N>::type type;
  };

  template<int N, int... Indices>
  struct make_indices_impl<N, indices<Indices...>, N> {
    typedef indices<Indices...> type;
  };
  
  template<int N>
  struct make_indices : make_indices_impl<0, indices<>, N> {};
  // end of stuff
  
  
  
  template<int I0, class F, int... I>
  constexpr std::array<decltype(std::declval<F>()(std::declval<int>())), sizeof...(I)>
  do_make(F f, indices<I...>)
  {
    return  std::array<decltype(std::declval<F>()(std::declval<int>())),
		       sizeof...(I)>{{ f(I0 + I)... }};
  }
  
  template<int N, int I0 = 0, class F>
  constexpr std::array<decltype(std::declval<F>()(std::declval<int>())), N>
  make(F f) {
    return do_make<I0>(f, typename make_indices<N>::type());
  }

} // namespace rowOffsetsUtils 


template<unsigned int D>
struct RowOffsets {
  inline RowOffsets() {
    int v[D];
    v[0]=0;
    for (unsigned int i=1; i<D; ++i)
      v[i]=v[i-1]+i;
    for (unsigned int i=0; i<D; ++i) { 
      for (unsigned int j=0; j<=i; ++j)
	fOff[i*D+j] = v[i]+j; 
      for (unsigned int j=i+1; j<D; ++j)
	fOff[i*D+j] = v[j]+i ;
    }
  }
  inline int operator()(unsigned int i, unsigned int j) const { return fOff[i*D+j]; }
  inline int apply(unsigned int i) const { return fOff[i]; }
  int fOff[D*D];
};


template<int D>  struct M {
 
  static constexpr int off0(int i) { return i==0 ? 0 : off0(i-1)+i;} 
  static constexpr int off2(int i, int j) { return j<i ? off0(i)+j : off0(j)+i; }
  static constexpr int off1(int i) { return off2(i/D, i%D);}

  static int off(int i) {
    static constexpr auto v = rowOffsetsUtils::make<D*D>(off1);
    return v[i];
  }


};


int myOff(int i) {
  return M<6>::off(i);
}

int oldOff(int i) {
    static RowOffsets<6> v;
    return v.apply(i);
}


#include<iostream>
int main() {
  RowOffsets<6> r;

  for (int i=0; i<6*6; ++i)
    std::cout << r.apply(i) << " " << M<6>::off(i) << std::endl;

}
