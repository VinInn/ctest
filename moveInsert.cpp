#include<vector>
#include<iostream>


struct B {
  virtual ~B(){}
  virtual B * clone() const=0;
  virtual void set(int)=0;
  virtual int get() const=0;
};

struct A : public B{
  int i=-2;
  A * clone() const { return new A(*this);}
  void set(int ii) {i=ii;}
  int get() const{ return i;}
};


struct P {
  P(){}
  explicit P(B* b) : a(b) {}
  P(const P&p) : a(p.a ? p.a->clone() : nullptr){}
  P(P&&p) : a(p.a){p.a=nullptr;}
  ~P(){delete a; a=nullptr;}
  P & operator=(const P&p) { if (a!=p.a) { delete a; a = p.a->clone();} return *this;}
  B *a=nullptr;
};


typedef std::vector<P> V;
typedef std::vector<std::vector<P>> VV;


typedef std::vector<std::vector<int>> VVI;


int move(V & res, V& v) {
  res.insert(res.end(),make_move_iterator(v.begin()),make_move_iterator(v.end()));
  return res.size();
}

int go() {
  //  VVI one(10,std::vector<int>(7,1));
  // VVI two(5,std::vector<int>(5,2));

  V one(10,P(new A)); one[0].a->set(7);
  V two(5,P(new A)); two[0].a->set(2);


  V res;
  int t =  move(res,one);
  std::cout << one.size() << " " << two.size() << " " << res.size() << std::endl;
  std::cout << one[0].a << " " << two[0].a << " " << res[0].a << std::endl;
  std::cout << one[0].a->get() << " " << two[0].a->get() << " " << res[0].a->get() << std::endl;

  /*
  std::cout << one.size() << " " << two.size() << " " << res.size() << std::endl;
  std::cout << one[0].size() << " " << two[0].size() << " " << res[0].size() << std::endl;
  res.insert(res.begin(),make_move_iterator(two.begin()),make_move_iterator(two.end()));
  std::cout << one.size() << " " << two.size() << " " << res.size() << std::endl;
  std::cout << one[0].size() << " " << two[0].size() << " " << res[0].size() << std::endl;

  VVI res2(make_move_iterator(res.begin()),make_move_iterator(res.end()));
  std::cout << res.size() << " " << res.size() << std::endl;
  std::cout << res[0].size() << " " << res2[0].size() << std::endl;

  VVI res3(std::move(res2));
  std::cout << res2.size() << " " << res3.size() << std::endl;
  */
  return t;
}


int main() {
  return go();
}
