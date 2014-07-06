#include<utility>
#include<tuple>
#include<iostream>
#include<cmath>

// pretty-print a tuple (from http://stackoverflow.com/a/6245777/273767 )
 
template<class Ch, class Tr, class Tuple, std::size_t... Is>
void print_tuple_impl(std::basic_ostream<Ch,Tr>& os,
                      const Tuple & t,
                      std::index_sequence<Is...>)
{
    using swallow = int[];
    (void)swallow{0, (void(os << (Is == 0? "" : ", ") << std::get<Is>(t)), 0)...};
}
 
template<class Ch, class Tr, class... Args>
auto operator<<(std::basic_ostream<Ch, Tr>& os, const std::tuple<Args...>& t)
   -> std::basic_ostream<Ch, Tr>&
{
    os << "(";
    print_tuple_impl(os, t, std::index_sequence_for<Args...>{});
    return os << ")";
}


template<typename F, std::size_t... Is>
void pmake_impl(F f, std::index_sequence<Is...>) {
    using swallow = int[];
    (void)swallow{0, (void(std::cout << f(Is)<<' '), 0)...};

}

template< std::size_t N, typename F, typename Indices = std::make_index_sequence<N> >
void pmake(F f) {
  pmake_impl(f,Indices());
}
 

template<typename F, std::size_t... Is>
constexpr auto make_impl(F f, std::index_sequence<Is...>)->std::array<int,std::index_sequence<Is...>::size()>  {
  return std::array<int,std::index_sequence<Is...>::size()>{{f(Is)...}};
}


template< std::size_t N, typename F, typename Indices = std::make_index_sequence<N> >
constexpr std::array<int,N> make(F f) {
  return make_impl(f,Indices());
}
 



  constexpr int D = 15;
  constexpr int off0(int i) { return i==0 ? 0 : off0(i-1)+i;} 
  constexpr int off2(int i, int j) { return j<i ? off0(i)+j : off0(j)+i; }
  constexpr int off1(int i) { return off2(i/D, i%D);}


int main()
{

  auto tuple = std::make_tuple(1,"hi",3.14);
  std::cout << tuple << '\n';

  std::cout << std::make_tuple(std::cos(1.3f),"hello",5/4) << std::endl;



  pmake<D*D>(off1);
  std::cout << std::endl;

  constexpr auto a = make<D*D>(off1);
  constexpr auto i = a[5];
  std::cout << a.size() << ":  " << i << " |  ";
  for (auto i : a) std::cout << i << ", ";
  std::cout << std::endl;



}
