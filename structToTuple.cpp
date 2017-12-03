#include <type_traits>
#include<tuple>
#include<cassert>

struct any_type {
  template<class T>
  constexpr operator T(); // non explicit
};

template<class T, typename... Args>
decltype(void(T{std::declval<Args>()...}), std::true_type())
test(int);

template<class T, typename... Args>
std::false_type
test(...);

template<class T, typename... Args>
struct is_braces_constructible : decltype(test<T, Args...>(0))
{
};



template<class T>
auto constexpr to_tuple(T&& object) noexcept {
    using type = std::decay_t<T>;
    if constexpr(is_braces_constructible<type, any_type, any_type, any_type, any_type>{}) {
      auto&& [p1, p2, p3, p4] = object;
      return std::make_tuple(p1, p2, p3, p4);
    } else if constexpr(is_braces_constructible<type, any_type, any_type, any_type>{}) {
      auto&& [p1, p2, p3] = object;
      return std::make_tuple(p1, p2, p3);
    } else if constexpr(is_braces_constructible<type, any_type, any_type>{}) {
      auto&& [p1, p2] = object;
      return std::make_tuple(p1, p2);
    } else if constexpr(is_braces_constructible<type, any_type>{}) {
      auto&& [p1] = object;
      return std::make_tuple(p1);
    } else {
        return std::make_tuple();
    }
}


template<class Tuple, std::size_t... Is>
constexpr bool check_tuple_impl(bool nop,
                      const Tuple & t,
                      std::index_sequence<Is...>)
{
    constexpr std::array<bool,std::index_sequence<Is...>::size()> a{(!std::is_pointer<std::decay_t<decltype(std::get<Is>(t))>>::value) ...};
    for (auto k : a) nop= nop&&k;
    return nop;
}
 
template<class... Args>
constexpr bool check_tuple(const std::tuple<Args...>& t)
{
 
    return check_tuple_impl(true,t, std::index_sequence_for<Args...>{});
}



int main() {
    {
      struct s {
        int p1;
        double p2;
      };

      auto t = to_tuple(s{1, 2.0});
      static_assert(std::is_same<std::tuple<int, double>, decltype(t)>{});
      assert(1 == std::get<0>(t));
      assert(2.0 == std::get<1>(t));
      assert(check_tuple(t));

     constexpr auto t2 = to_tuple(s{1, 2.0});
     static_assert(check_tuple(t2));

    }

    {
      struct s {
        struct nested { } p1;
        int p2;
        int p3;
        s* p4;
      };

      auto t = to_tuple(s{s::nested{}, 42, 87, nullptr});
      static_assert(std::is_same<std::tuple<s::nested, int, int, s*>, decltype(t)>{});
      assert(42 == std::get<1>(t));
      assert(87 == std::get<2>(t));
      assert(nullptr == std::get<3>(t));
      assert(!check_tuple(t));


      constexpr auto t2 = to_tuple(s{s::nested{}, 42, 87, nullptr});
      static_assert(!check_tuple(t2)); 

      assert(42 == std::get<1>(t2));
      assert(87 == std::get<2>(t2));
      assert(nullptr == std::get<3>(t2));
      assert(!check_tuple(t2));

      assert(std::is_pointer<std::decay_t<decltype(std::get<3>(t2))>>::value);
      assert(!std::is_pointer<std::decay_t<decltype(std::get<1>(t2))>>::value);

    }
}
