#include "nativeVector.h"

#include<type_traits>
#include<tuple>
#include<array>

template<typename T>
struct Less {
  auto operator()(T x, T y)->decltype(x<y) { return x<y;}
};

template<typename T>
class DecisionNode {
public:
  using index_type = unsigned short; 
  using S =  decltype(nativeVector::VType<T>::elem(T()));
  using B = typename std::result_of<Less<T>(T,T)>::type;
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





template<typename T, int N>
class DecisionTree {
public:
  
  using Node = DecisionNode<T>;
  using index_type = typename Node::index_type;
  using B = typename Node::B;
  using S = typename Node::S;
  using SubTree=DecisionTree<T,N-1>;
  
  static constexpr unsigned int ret_size=2*SubTree::ret_size;
  using return_type = std::array<B,ret_size>;
  static constexpr unsigned int size=1+2*SubTree::size;


  DecisionTree(S const * ic, index_type const * ii) :
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


template<typename T>
class DecisionTree<T,1> {
public:
  using Node = DecisionNode<T>;
  using B = typename Node::B;
  using S = typename Node::S;
  using index_type = typename Node::index_type;

  static constexpr unsigned int ret_size=2;
  using return_type = std::array<B,ret_size>;
  static constexpr unsigned int size=1;


  DecisionTree(S const * ic, index_type const * ii) : node(ic[0],ii[0]){}
  
  void operator()(T const * x, B prev, B * ret) const {
    node(x,prev,ret);
  }  
  return_type operator()(T const * x) const {
    return node(x);
  }
  
private:
  DecisionNode<T> node;

};


template<typename T>
class DecisionTree<T,0> {
public:
  using Node = DecisionNode<T>;
  using B = typename Node::B;
  using S = typename Node::S;
  using index_type = typename Node::index_type;

  static constexpr unsigned int ret_size=1;
  static constexpr unsigned int size=0;


  DecisionTree(S const *, index_type const *) {}
  
  void operator()(T const *, B, B *) const {}
  
};
  


#include<iostream>

int main() {

  using namespace nativeVector;

  auto ts = DecisionTree<int, 4>::size;
  auto rs = DecisionTree<int, 4>::ret_size;
  using index_type = DecisionTree<int, 4>::index_type;

  index_type ind[ts]={0};
  int cuts[ts];
  
  DecisionTree<int, 4> tree(cuts,ind);

  int val[1]={4};
  auto c = tree(val);

  for ( auto b : c) std::cout << b;
  std::cout << std::endl;
  
  short scuts[ts];
  DecisionTree<SVect, 4> treev(scuts,ind);
  short w[rs]; w[0]=1; for (auto i=rs-rs+1; i<rs; ++i) w[i]=w[i-1]+1;
  
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
  

  return 0;
}
