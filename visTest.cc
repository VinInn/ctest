namespace s __attribute__((visibility("default"))) {
  template <class T>
    class vector {
  public:
    vector() { }
  };
  
  template <class T>
  inline  void foo(T t) {
  }
}


namespace h {
  template <class T>
    class vector {
  public:
    vector() { }
  };
  template <class T>
  inline  void foo(T t) {
  }
}

class A {
public:
  A() { }
  s::vector<int> vs;
  h::vector<int> vh;

};

s::vector<A> v;

int main() {
  A a;
  s::foo(a);
  s::foo(v);
  s::foo(a.vs);
  h::foo(a);
  h::foo(v);
  h::foo(a.vh);

  return 0;
}

