#ifdef NOFINAL
#define final
#endif
struct A {
  virtual ~A(){}
  virtual int a() const final  { return m_a;}
  int b() const { return m_b;}
  virtual int c() const { return m_c;}
  virtual int operator()() const final { return m_a;}

  int geta() const;
  int getb() const;
  int getc() const;
  int get() const;

  int m_a;
  int m_b;
  int m_c;

};


int A::get() const { return (*this)();}
int A::geta() const { return a();}
int A::getb() const { return b();}
int A::getc() const { return c();}

struct B final : public A {

  int got() const;
  int gota() const;
  int gotb() const;
  int gotc() const;

};
 

int B::got() const { return (*this)();}
int B::gota() const { return a();}
int B::gotb() const { return b();}
int B::gotc() const { return c();}

