template<typename T>
class A {
public:
  explicit A (T const & it) : t(it){}

  T const & data() const;

  T t;

};


template<typename T>
T const & A<T>::data() const {
    return t;
  }


extern template class A<int>;

template class A<int>;



int main() {

  A<int> a(3);

  return a.data();


}
