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

#include <cstddef>
#include <iostream>
#include <utility>
#include <type_traits>

template<class T, std::size_t N>
struct carray {
  T data[N];
  constexpr const T& operator[](std::size_t i) { return data[i]; }
  T& operator[](std::size_t i) { return data[i]; }
  constexpr std::size_t size() { return N; }
  T* begin() { return data; }
  T* end() { return data + N; }
  const T* begin() const { return data; }
  const T* end() const { return data + N; }
};

template<int I0, class F, int... I>
constexpr carray<decltype(std::declval<F>()(std::declval<int>())), sizeof...(I)>
do_make(F f, indices<I...>)
{
  return carray<decltype(std::declval<F>()(std::declval<int>())),
    sizeof...(I)>{{ f(I0 + I)... }};
}

template<int N, int I0 = 0, class F>
constexpr carray<decltype(std::declval<F>()(std::declval<int>())), N>
make(F f) {
  return do_make<I0>(f, typename make_indices<N>::type());
}

constexpr float g(int i) { return -i + 0.5; }


/* int n = N;
    union {
      float numlog;
      int x;
    } tmp;
    tmp.x = 0x3F800000; //set the exponent to 0 so numlog=1.0
    int incr = 1 << (23-n); //amount to increase the mantissa
    int p=std::pow(2.,n);
    lookup_table = new float[p];
    for(int i=0;i<p;++i)
      {
	lookup_table[i] = log2(tmp.numlog); //save the log value
	tmp.x += incr;
      }
*/

#include <cmath>
template<int N>
struct logTable {
  static constexpr int incr() { return 1 << (23-N); }
  static constexpr int size() { return std::pow(2.,N);}
  static constexpr float value(int i) {
    union {
      float numlog;
      int x;
    } tmp;
    return (tmp.x =  0x3F800000 + i*incr()) >0 ? log2(tmp.numlog) : 0;
  }
};
int main()
{
  constexpr auto v = make<logTable<16>::size()>(logTable<16>::value);
  /*
  constexpr auto v = make<256>(g); // OK
  constexpr auto e1 = v[1]; // OK
  for (auto i : v) std::cout << i << std::endl;
  */
  return 0;
}

