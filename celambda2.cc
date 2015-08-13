template<typename F>
constexpr int foo(F f, int q) { return f(q);}

constexpr int bar(int q) {
 constexpr auto mask = foo([](int i){ return i;},42);
 return q&mask;
}

constexpr int bar3(int q) {
 return q&foo([](int i){ return i;},42);
}


constexpr auto foo2(int i) { return i;}
 
constexpr int bar2(int q) {
 constexpr auto mask = foo(foo2,42);
 return q&mask;
}

constexpr int k3 = bar3(1);
constexpr int k2 = bar2(1);

