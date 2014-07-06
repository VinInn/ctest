#include<vector>
#include<tuple>


template<typename A, typename B>
struct MultiV {


  std::tuple<std::vector<A>,std::vector<B>> vs;
};


MultiV<int,float> mif;


template<typename A>
struct h1 {
  std::vector<A> operator()();
};

template<typename A>
auto h(A)->std::vector<A>;




template<typename ...Args>
auto func(Args... args) -> decltype(std::make_tuple(h(args)...));


/*
template<typename HEAD, typename ... Tail>
struct TV {
  typedef typename TV<Tail...> tn;
  typedef std::tuple<std::vector<HEAD>,std::vector<Tail>...> type;
};

template<typename HEAD, typename Tail>
struct TV {
  typedef std::tuple<std::vector<HEAD>,std::vector<Tail>> type;
};
*/


template<typename... Args>
struct MultiW {
  typedef std::tuple<std::vector<Args>...> TUPLE;
  static constexpr std::size_t SIZE = std::tuple_size<TUPLE>::value;
  static constexpr std::size_t size() { return SIZE;}

  template<int N>
  auto get(int i)->typename std::tuple_element<N,TUPLE>::type::const_reference  const {
    return std::get<N>(vs)[i];
  }

  std::tuple<std::vector<Args>...> vs;

};

MultiW<int,float> mof;



#include<typeinfo>
#include<iostream>
int main() {
  std::cout << mof.size() << std::endl;
  int i;
  std::get<0>(mof.vs).push_back(1);
  std::cout << mof.get<0>(0) << std::endl;
  std::get<1>(mof.vs).push_back(3.14);
  std::cout << mof.get<1>(0) << std::endl;

  typedef std::result_of<h1<int>()>::type H;
  std::cout << typeid(H).name() << std::endl;
  std::cout << typeid(mof.vs).name() << std::endl;
  // typedef std::result_of<mof.get<1>(int)>::type A;
  // std::cout << typeid(A).name() << std::endl;

  return 0;

}
