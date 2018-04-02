#include <vector>
#include <iostream>
struct V : std::vector<V>{};
int main()
{
  V v;
  v.emplace_back();

  std::cout << &v.front() << std::endl;
  std::cout << v.size() << std::endl;

  v.swap(v.front()); 

  std::cout << &v.front() << std::endl;
  std::cout << v.size() << std::endl;
 
  return v.size();
}

