#include<type_traits>
#include<typeinfo>
#include<iostream>


template<typename T, int N>
struct ExtVecTraits {
  typedef T __attribute__( ( vector_size( N*sizeof(T) ) ) ) type;
};

template<typename T>
struct ExtVecTraits<T,0> {
  typedef T type;
};

template<typename T>
struct ExtVecTraits<T,1> {
  typedef T type;
};


template<typename V>
struct scalar {
  //  typedef typename std::remove_reference<decltype(v[0])>::type type;
};

template<>
struct scalar<float> {
  typedef float type;
};



template<typename T, int N> using ExtVec =  typename ExtVecTraits<T,N>::type;


template<size_t N> 
struct number_bysize {
};

template<>  
struct number_bysize<32> {
  typedef number_bysize<64> scalar_type;
  typedef float binary;
  typedef signed int signed_int;
  typedef unsigned int unsigned_int;
  union number {
    number(){}
    number(binary n) : f(n){}
    number(signed_int n) : i(n){}
    number(unsigned_int n) : u(n){}
    binary f;
    signed_int i;
    unsigned_int u;
  };

  using number_f = number;

  template<int N> using binaryVector =  typename ExtVecTraits<binary,N>::type;
  template<int N> using signedVector =  typename ExtVecTraits<signed_int,N>::type;
  template<int N> using unsignedVector =  typename ExtVecTraits<unsigned_int,N>::type;

  template<int N>
  union vector {
    vector(){}
    vector(binaryVector<N> n) : f(n){}
    vector(signedVector<N> n) : i(n){}
    vector(unsignedVector<N> n) : u(n){}
    binaryVector<N> f;
    signedVector<N> i;
    unsignedVector<N> u;
  };

};

template<>
struct number_bysize<64> {
  typedef number_bysize<64> scalar_type;
  typedef double binary;
  typedef signed long long signed_int;
  typedef unsigned long long  unsigned_int;
  union number {
    number(){}
    number(binary n) : f(n){}
    number(signed_int n) : i(n){}
    number(unsigned_int n) : u(n){}
    binary f;
    signed_int i;
    unsigned_int u;
  };
  using number_d = number;

  template<int N> using binaryVector =  typename ExtVecTraits<binary,N>::type;
  template<int N> using signedVector =  typename ExtVecTraits<signed_int,N>::type;
  template<int N> using unsignedVector =  typename ExtVecTraits<unsigned_int,N>::type;

  template<int N>
  union vector {
    vector(){}
    vector(binaryVector<N> n) : f(n){}
    vector(signedVector<N> n) : i(n){}
    vector(unsignedVector<N> n) : u(n){}
    binaryVector<N> f;
    signedVector<N> i;
    unsignedVector<N> u;
  };


};

template<typename T, class Enable=void> struct number_type; // = number_bysize<8*sizeof(T)>;

template<typename T>
struct number_type<T, typename std::enable_if<std::is_arithmetic<T>::value >::type> 
 : public number_bysize<8*sizeof(T)>{};


template<typename V>
struct number_type<V, typename std::enable_if<std::is_compound<V>::value >::type> {
  static constexpr size_t  floatSize = sizeof(V)/4;
  static constexpr size_t  doubleSize = sizeof(V)/8;
  using n_float = number_type<float>;
  using n_double = number_type<double>;
  static constexpr bool is_f = std::is_same<V,n_float::binaryVector<floatSize>>::value;
  static constexpr bool is_d = std::is_same<V,n_double::binaryVector<doubleSize>>::value;
  using number_f = n_float::vector<floatSize>;
  using number_d = n_double::vector<doubleSize>;
  using number = typename std::conditional<is_f,number_f,number_d>::type;
  using scalar_type = typename std::conditional<is_f,n_float,n_double>::type;
};




int main() {

//std::cout << typeid(std::make_unsigned<float>::type).name() << std::endl;
  number_bysize<32>::number n32;
  number_bysize<64>::number n64;

  number_type<double>::number dd;

  using int32x8 = number_type<float>:: signedVector<8>;
  using v32x8 = number_type<float>::vector<8>;

  v32x8 v1;

  using b32x8 = number_type<float>::binaryVector<8>;
  using b64x4 = number_type<double>::binaryVector<4>;

  number_type<b64x4>::number_d nd4;
  number_type<b64x4>::number_f nf8;
  number_type<b64x4>::number n4;

  std::cout << std::boolalpha;
  std::cout << std::is_fundamental<b32x8>::value << std::endl;
  std::cout << std::is_object<b32x8>::value << std::endl;
  std::cout << std::is_compound<b32x8>::value << std::endl;
  std::cout << std::is_floating_point<b32x8>::value << std::endl;
  std::cout << std::is_arithmetic<b32x8>::value << std::endl;
  std::cout << std::is_scalar<b32x8>::value << std::endl;
  std::cout << std::is_array<b32x8>::value << std::endl;
  std::cout << number_type<b64x4>::is_f  << std::endl;
  std::cout << number_type<b64x4>::is_d  << std::endl;
  typedef typename std::remove_all_extents<b32x8>::type Type;
  std::cout << "underlying type: " << typeid(Type).name() << '\n';
  std::cout << "underlying type: " << typeid(b64x4).name() << '\n';


  std::cout << typeid(n32.i).name() << std::endl;
  std::cout << typeid(n32.u).name() << std::endl;
  std::cout << typeid(dd.u).name() << std::endl;
  std::cout << typeid(v1.u).name() << std::endl;
  std::cout << typeid(nd4.u).name() << std::endl;
  std::cout << typeid(nf8.u).name() << std::endl;
  std::cout << typeid(n4.u).name() << std::endl;

  std::cout << typeid(number_type<double>::unsigned_int).name() << std::endl;
  std::cout << sizeof(number_type<double>::unsigned_int) << std::endl;
  std::cout << typeid(unsigned long long).name() << std::endl;
  std::cout << sizeof(int32x8) << std::endl;
  std::cout << sizeof(b64x4) << std::endl;
return 0;

}
