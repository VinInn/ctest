#include<cstdlib>
#include<cstdio>
void __attribute__((visibility("default"))) bar() { printf("hi\n");;}

struct A {
   A();
   ~A();

  static A & get() {
    static A a; return a;
 }

};

struct B {
   inline B();
   inline ~B();
};



namespace h {
  inline void foo() {
      bar();
  }

  inline void __attribute__((visibility("default")))  fooV() { bar();}


template<typename T>
inline T & get() __attribute__((visibility("default")));
template<typename T>
 T & get() {
    static T t;
    return t; 
 }


template<typename T>
inline  T & getH() {
    static T t;
    return t;
 }


template<typename T>
T hello() {
  return T();
}

}



namespace v __attribute__((visibility("default"))) {
  inline void foo() {
      bar();
  }

  inline void __attribute__((visibility("default")))  fooV() { bar();}

template<typename T>
inline T & get() {
    static T t;
    return t;
 }


template<typename T>
T hello() {
  return T();
}

}


  A::A() {bar(); h::foo(); h::fooV();}
  A::~A() {bar(); v::foo(); v::fooV();}


A k0 = A::get();
A k1 = h::get<A>();
A k2 = h::getH<A>();
A k3 = v::get<A>();
A k4 = h::hello<A>();
A k5 = v::hello<A>();


  B::B() {bar(); h::foo(); h::fooV();}
  B::~B() {bar(); v::foo(); v::fooV();}

B j1 = h::get<B>();
B j2 = h::getH<B>();
B j3 = v::get<B>();
B j4 = h::hello<B>();
B j5 = v::hello<B>();



