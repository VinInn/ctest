#include "nativeVector.h"

#include<type_traits>
#include<tuple>
#include<array>



class DecisionNode {
public:
  using index_type = unsigned short; 
  using S = short;
  using B = nativeVector::SVect;
  using T = nativeVector::SVect;
  using return_type = std::array<B,2>;


  DecisionNode( S ic, index_type ii) : c(ic), i(ii){}
  
  void operator()(T const * x, B prev, B * ret) const {
     auto r = x[i]<c;
     ret[0]=r&prev; ret[1]=(!r)&(prev);
  }

  return_type operator()(T const * x) const {
    auto r = x[i]<c;
    return return_type{r,!r};
  }

  
  S c;
  index_type i;
};





template<int N>
class DecisionTreeImpl {
public:
  
  using Node = DecisionNode;
  using index_type = Node::index_type;
  using B = Node::B;
  using S = Node::S;
  using T = Node::T;
  using SubTree=DecisionTreeImpl<N-1>;
  
  static constexpr unsigned int ret_size=2*SubTree::ret_size;
  using return_type = std::array<B,ret_size>;
  static constexpr unsigned int size=1+2*SubTree::size;


  DecisionTreeImpl(S const * ic, index_type const * ii) :
    node(ic[0],ii[0]),
    left(ic+1,ii+1),
    right(ic+1+SubTree::size,ii+1+SubTree::size)
  {}

  
  void operator()(T const * x, B prev, B * ret) const {
    typename Node::return_type d; node(x, prev,&d[0]);
    left(x,d[0],&ret[0]); right(x,d[1],&ret[ret_size/2]);
  }

  return_type operator()(T const * x) const {
    auto d = node(x);
    return_type ret;
    left(x,d[0],&ret[0]); right(x,d[1],&ret[ret_size/2]);
    return ret;
  }

  Node node;
  SubTree left;
  SubTree right;

};


template<>
class DecisionTreeImpl<1> {
public:
  using Node = DecisionNode;
  using B = Node::B;
  using S = Node::S;
  using T = Node::T;
  using index_type = typename Node::index_type;

  static constexpr unsigned int ret_size=2;
  using return_type = std::array<B,ret_size>;
  static constexpr unsigned int size=1;


  DecisionTreeImpl(S const * ic, index_type const * ii) : node(ic[0],ii[0]){}
  
  void operator()(T const * x, B prev, B * ret) const {
    node(x,prev,ret);
  }  
  return_type operator()(T const * x) const {
    return node(x);
  }
  
private:
  DecisionNode node;

};


template<int N>
class DecisionTree {
public:
  
  using DT = DecisionTreeImpl<N>;
  using Node = DecisionNode;
  using index_type = Node::index_type;
  using B = Node::B;
  using S = Node::S;
  using T = Node::T;

  static constexpr unsigned int ret_size= DT::ret_size;
  static constexpr unsigned int size= DT::size;


  
  DecisionTree(S const * ic, index_type const * ii, S const * iw) :
    dt(ic,ii){std::copy(iw, iw+DT::ret_size, w);}

  T operator()(T const * x) const {
    T res={0};
    auto v = dt(x);
    int i=0; for ( auto & b : v ) b&=w[i++];
    for ( auto b : v) res|=b;
    return res;
  }
  
private:
  

  DT dt;

  S w[DT::ret_size];
  
};


#include<iostream>

int main() {

  using namespace nativeVector;

  auto ts = DecisionTree<4>::size;
  auto rs = DecisionTree<4>::ret_size;
  using index_type = DecisionTree<4>::index_type;

  index_type ind[ts]={0};

  
  short scuts[ts];
  DecisionTreeImpl<4> treev(scuts,ind);
  short w[rs]; w[0]=1; for (auto i=rs-rs+1; i<rs; ++i) w[i]=w[i-1]+1;
  DecisionTree<4> dtree(scuts,ind,w);

  
  std::cout << "dt size " << sizeof(treev) << std::endl;

  SVect zero={0};
  SVect sv[1]={zero+4};
  SVect res = zero;
  auto v = treev(sv);
  for ( auto b : v) std::cout << b <<' ';
  std::cout << std::endl;
  int i=0; for ( auto & b : v ) b&=w[i++];
  for ( auto b : v) std::cout << b <<' ';
  std::cout << std::endl;
  for ( auto b : v) res|=b;
  std::cout << res << std::endl;
  
  std::cout << dtree(sv) << std::endl;

  
  return 0;
}
