template<typename T>
struct A {
  typedef T  type;
};

template<typename T>
typename A<T>::type
inline add(typename A<T>::type x, typename A<T>::type y){
return x+y;
}

typedef A<float>::type A4;


A4 addf(A4 x, A4 y) {
return add(x,y);
}

A4 sum(A4 x, A4 y) {
return x+y;
}





