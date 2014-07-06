#include<vector>
struct bar {
   static void * foo();
   virtual void * getFoo() const { return bar::foo();}
};

struct base {
   base();
   virtual ~base();
   virtual base * clone() const=0;
   static void *	qw();
   virtual void * hi() const { return base::qw();}
};

struct foo {
   foo();
   ~foo();
    foo(foo const&);
   int v;

static foo bf(int h) {
  foo f; f.v=h; return f;
}
virtual foo * clone() const {
  return new foo(*this);
}
  virtual int get() const { return v;}

  static void * qw();
   virtual void	* hi() const { return foo::qw();}

};


inline
foo bfoo(int h) {
  foo f; f.v=h; return f;
}

int add(std::vector<foo*> & v, foo * f) {
  v.push_back(f);
  return v.size();
}
