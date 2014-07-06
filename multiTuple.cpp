#include <tuple>
#include <utility>

template<typename T, int J, typename Tuple, std::size_t... I>
T tupleRange_impl(Tuple& a, std::index_sequence<I...>) {
   return  T{std::get<J+I>(a) ...};
}


template<typename T, int J, typename Tuple>
T tupleRange(Tuple & in) {
  return  tupleRange_impl<T,J>(in,std::make_integer_sequence<std::size_t,std::tuple_size<T>::value>{});
}


#include<iostream>
int main() {

   using TA = std::tuple<int&,int&>;
   using TB = std::tuple<float&,int&>;

   using C = decltype(std::tuple_cat(std::declval<TA>(),std::declval<TB>()));
   
   int i=3; float f = 3.14;

   C c = std::tie(i,i,f,i);
   
   TA a = tupleRange<TA,0>(c); 
   TB b = tupleRange<TB,2>(c);

   std::get<0>(b) +=1;
 

   std::cout << std::get<2>(c) << std::endl;

   return 0;
}
