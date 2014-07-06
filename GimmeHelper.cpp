#include<vector>

struct Gimme {
  virtual ~Gimme(){}
  virtual gimme(void *& t,index i, prodId j, Event const & ev) const=0;
};


vector<Gimme *> gimmeRegister;

template<typename T>
struct GimmeVector {
  virtual gimme(void *& t,index i, prodId j, Event const & ev) const {
    vector<T> const * v;  ev.get(j,v);
    t = &(*v)[i];
  }

};


template<typename T>
struct Ref {
  prodId j;
  Index i;
  gimmeId g;

  void get(Event const & ev) {
    gimmeRegister[g]->gimme(pts,i, j,ev); 
  }
 

  void * ptr;
};
