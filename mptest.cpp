namespace details {

  typedef char (& no_tag )[1]; // type indicating FALSE
  typedef char (& yes_tag)[2]; // type indicating TRUE
  
  // Definitions for the following struct and function templates are
  // not needed; we only require the declarations.
  template <typename T, void (T::*)(T&)>  struct swap_function;
  template <typename T> no_tag  has_swap_helper(...);
  template <typename T> yes_tag has_swap_helper(swap_function<T, &T::swap> * dummy);

  
  template<typename T>
  struct has_swap_function
  {
    static bool const value = 
      sizeof(has_swap_helper<T>(0)) == sizeof(yes_tag);
  };

  template <typename T, typename V>  struct value_type;
  template <typename T> no_tag  has_value_helper(...);
  template <typename T> yes_tag has_value_helper(value_type<T, typename T::value_type> * dummy);

  
  template<typename T>
  struct has_value_type
  {
    static bool const value = 
      sizeof(has_value_helper<T>(0)) == sizeof(yes_tag);
  };


}

#include<vector>
#include<iostream>


struct A {};

int main() {

  std::cout << details::has_swap_function<std::vector<A> >::value << std::endl;
  std::cout << details::has_swap_function<int >::value << std::endl;
  std::cout << details::has_swap_function<A >::value << std::endl;

  std::cout << details::has_value_type<std::vector<A> >::value << std::endl;
  std::cout << details::has_value_type<int >::value << std::endl;
  std::cout << details::has_value_type<A >::value << std::endl;

  return 0;
}
