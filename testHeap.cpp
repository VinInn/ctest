#include<algorithm>
#include<iostream>
#include<vector>
#include<tuple>



int main() {

  using A = std::pair<int,int>;
  using C = std::vector<A>;


  C c;
  c.emplace_back(9,-1);
  std::push_heap(c.begin(),c.end(),[](A & a, A&b) { return a.first < b.first;});
#ifdef ADD8
  c.emplace_back(8,-1);
  std::push_heap(c.begin(),c.end(),[](A & a, A&b) { return a.first < b.first;});
#endif
  for (int i=0; i<10; ++i) {
    c.emplace_back(i,0);
    std::push_heap(c.begin(),c.end(),[](A & a, A&b) { return a.first < b.first;});
  }

   c.emplace_back(9,1);
   std::push_heap(c.begin(),c.end(),[](A & a, A&b) { return a.first < b.first;});
  
  std::cout << c.front().first << ' ' << c.front().second << std::endl;
  std::pop_heap(c.begin(),c.end(),[](A & a, A&b) { return a.first < b.first;});
  c.pop_back();
  std::cout << c.front().first << ' ' << c.front().second << std::endl;
  std::pop_heap(c.begin(),c.end(),[](A & a, A&b) { return a.first < b.first;});
  c.pop_back();
  std::cout << c.front().first << ' ' << c.front().second << std::endl;
  std::pop_heap(c.begin(),c.end(),[](A & a, A&b) { return a.first < b.first;});
  c.pop_back();
  std::cout << c.front().first << ' ' << c.front().second << std::endl;
  std::pop_heap(c.begin(),c.end(),[](A & a, A&b) { return a.first < b.first;});
  c.pop_back();
  std::cout << c.front().first << ' ' << c.front().second << std::endl;


  return 0;
}
