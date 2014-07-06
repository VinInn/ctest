#include<vector>
#include <tuple>
#include <iostream>
#include <array>
#include <utility>
#include <type_traits>


template<typename T>
struct Vect {
 using value = typename std::remove_reference<T>::type;
 using ref = typename std::add_lvalue_reference<T>::type;
 using CREF = Vect<value const &>;
 using REF = Vect<ref>;


  Vect() {}
  Vect(T ix, T iy, T iz, int k=0) : x(ix),y(iy),z(iz){}
  template<typename V>
  Vect(V v) : x(v.x), y(v.y), z(v.z) {}
  Vect(Vect const & v) : x(v.x), y(v.y), z(v.z) {}
  template<typename V>
  Vect& operator=(V v) { x=v.x; y=v.y; z=v.z; return *this; }
  Vect& operator=(Vect const & v) { x=v.x; y=v.y; z=v.z; return *this; }

  T x,y,z;
};


template<typename T, typename ... Args>
class Soa {
public:
   using CREF = typename T::CREF;
   using REF = typename T::REF;


   using Data = std::tuple<std::vector<Args>...>;

   
   template<std::size_t... I>
   void resize(unsigned int n, std::index_sequence<I...>) { 
     using swallow = int[];
     (void)swallow{0, ((void)(std::get<I>(data).resize(n)),0)...};
   }
   
  
   explicit Soa(unsigned int n) : m_n(n){
     resize(n,std::index_sequence_for<Args...>{});
   }

  auto size() const { return m_n;}

  template<typename V, std::size_t... I>
   V t2r_impl(int j, std::index_sequence<I...>) {
   return V(std::get<I>(data)[j] ...); 
 }

  REF operator[](int j) {
   return t2r_impl<REF>(j,std::index_sequence_for<Args...>{});
 }

  CREF operator[](int j) const {
   return t2r_impl<CREF>(j,std::index_sequence_for<Args...>{});
 }



  Data data;
  unsigned int m_n;
};

template<typename Array, std::size_t... I>
Vect<float &> a2v_impl(Array& a, int j, std::index_sequence<I...>) {
   return Vect<float &> (a[I][j] ...); // return v;
}

template<typename Tuple, std::size_t... I>
Vect<float &> t2v_impl(Tuple& a, int j, std::index_sequence<I...>) {
   return Vect<float &> (std::get<I>(a)[j] ...); // return v;
}


template<typename ... Args>
Vect<float &> t2v(std::tuple<Args...>& a, int j)
{
    return t2v_impl(a, j, std::index_sequence_for<Args...>{});
}

template<typename T, std::size_t N, typename Indices = std::make_index_sequence<N>>
Vect<float &> a2v(std::array<T, N>& a, int j)
{
    return a2v_impl(a, j, Indices());
}

#include<iostream>

int main() {

  using V = std::vector<float>;
  using SoaV = std::array<V,3>;
  using SoaN = std::tuple<V,V,V>;

  SoaV s = {V(10),V(10),V(10)};
  SoaN s2{V(10),V(10),V(10)};


  Vect<float &> v0 = a2v(s, 3);
  Vect<float &> v2 = t2v(s2, 5);
  v0.x=-9.; v2.y=v0.x;

  Soa<Vect<float>, float,float,float,int> soaI(30);
  std::cout << soaI.size() << std::endl;

  Vect<float &> vk = soaI[23];
  vk.x=3.14;

  soaI[11] = v0;

  

  return v0.x*v2.y;
}
