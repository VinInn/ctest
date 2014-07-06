struct Config_Foo {
  // Config_Foo(){}
  template<typename CI>
  constexpr Config_Foo(CI const & ci) : i(ci.i), f(ci.f){}

  const int i;
  const float f;

};


struct Foo : public Config_Foo {
  typedef Config_Foo Config;
  constexpr Foo(Config_Foo const & c) : Config_Foo(c){}

};

namespace {

  struct MyConfig_Foo {
    int a=2;

    const int i=3;
    const float f=4.14f;
    
  };
  
  MyConfig_Foo a;
  Foo::Config b(a);
  inline Foo & foofact() {
    static Foo foo(b);
    return foo;
  }
  
}

#include<cassert>
int main() {
  auto const & foo=foofact();
  assert(foo.i==3);
  assert(foo.f==4.14f);

}

