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

  using B = typename std::result_of<Less<T>(T,T)>::type;
  using return_type = std::array<B,2>;
  
  void operator()(T x, B prev, B * ret) const {
     auto r = x<c;
     ret[0]=r&prev; ret[1]=(!r)&(prev);
  }

  return_type operator()(T x) const {
    auto r = x<c;
    return return_type{r,!r};
  }

  
  T c;
  
};





template<typename T, int N>
class DecisionTree {
public:
  using Node = DecisionNode<T>;
  using B = typename Node::B;
  using SubTree=DecisionTree<T,N-1>;
  
  static constexpr unsigned int size=2*SubTree::size;
  using return_type = std::array<B,size>;

  void operator()(T x, B prev, B * ret) const {
    typename Node::return_type d; node(x, prev,&d[0]);
    left(x,d[0],&ret[0]); right(x,d[1],&ret[size/2]);
  }

  return_type operator()(T x) const {
    auto d = node(x);
    return_type ret;
    left(x,d[0],&ret[0]); right(x,d[1],&ret[size/2]);
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

  static constexpr unsigned int size=2;
  using return_type = std::array<B,size>;

  void operator()(T x, B prev, B * ret) const {
    node(x,prev,ret);
  }  
  return_type operator()(T x) const {
    return node(x);
  }
  
private:
  DecisionNode<T> node;

};
  

#include<iostream>

int main() {

  DecisionTree<int, 4> tree;

  auto c = tree(4);

  for ( auto b : c) std::cout << b;
  std::cout << std::endl;
  
  return 0;
}
