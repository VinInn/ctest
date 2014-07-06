#include<vector>
#include<algorithm>
#include<iostream>

typedef unsigned int  id_type;
typedef unsigned int  size_type;

struct Item {
  Item(id_type i=0, int io=-1, size_type is=0) : id(i), offset(io), size(is){}
  id_type id;
  int offset;
  size_type size;
  bool operator<(Item const &rh) const { return id<rh.id;}
  operator id_type() const { return id;}
};

struct cmp10 {
  bool operator()(id_type i1, id_type i2) const {
    return i1/10 < i2/10;
  }
  };


std::pair<id_type,cmp10> bha() {
  return std::make_pair<10,cmp10());
}

template<typename F>
struct undoPair {
  undoPair(F& ff) :f(ff){}
  F & f;
  F::result_type operator()(std::pair<A,B> const & p) {
    return f(p.first,p.second);
  }
};


int main() {
  
  unsigned int u[] = {10,11,12,30,41,44,57};
  std::vector<Item> v(7);
  typedef std::vector<Item>::const_iterator Iter;
  std::copy(u,u+7,v.begin());
  std::pair<Iter,Iter> r =
    std::equal_range(v.begin(),v.end(),Item(11));
  std::cout << r.second-r.first << std::endl;
  r =
    std::equal_range(v.begin(),v.end(),30,cmp10());
  std::cout << r.second-r.first << std::endl;
  r =
    std::equal_range(v.begin(),v.end(),40,cmp10());
  std::cout << r.second-r.first << std::endl;
  r = undoPair( bha
  return 0;

}
