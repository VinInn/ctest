#include<vector>

struct B;

struct A{
  int i;
  B* b;
  A();
  ~A();
  A(A const &);
  A(A&&) noexcept;
  bool ok() const;
  void mod();
};

A getA();

int got() {
  A a = getA();
  if (a.ok()) a.mod();
  return a.i;
}

typedef std::vector<A> VA;

A foo(VA const & va, int i) {
  return std::move(va[i]);
} 

VA foo(VA const & va) {
  VA v;
  for (auto & a : va) 
    if (a.ok()) v.push_back(std::move(a));
  return v;
}


A bar(VA & va, int i) {
  return std::move(va[i]);
} 

VA bar(VA & va) {
  VA v;
  for (auto & a : va) 
    if (a.ok()) v.push_back(std::move(a));
  return v;
}
/*
void eee(VA & va);

int fun(VA & va) {

  eee(va);
  VA v = bar(va);
  if (v[1].i>0) v.resize(1);

  return v.size();
}
*/
