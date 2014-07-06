#include<type_traits>

template<typename T> 
struct V {
  using value = typename std::remove_reference<T>::type;
  using ref = typename std::add_lvalue_reference<T>::type;

  V(ref ix) : xi(ix){}

  value x() const { return xi;}
  ref x() { return xi;}

  T xi;

};




#include<iostream>

template<typename V> void print(V const & v)  { std::cout << v.x() << std::endl;} 

int main() {

  float z = 0;
  V<float> vf(z); vf.x()=3; print(vf);

  V<float&> vr(vf.x());  print(vr); vr.x()=-2; print(vf);

  return 0;

}
