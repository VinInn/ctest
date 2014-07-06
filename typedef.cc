
template<class T>
class S {

public:

  inline void s(INT i) { m_s=i;}

private:

  typedef T INT;

  INT m_s;

};


void f() {

  S<int> s;
  s.s(2);

}
