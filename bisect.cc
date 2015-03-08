#include<vector>

using Iter=std::vector<int>::iterator;

// using Iter=unsigned int;

Iter bisect(Iter a, Iter b) {
  return a+(b-a)/2;
}

Iter bisect2(Iter a, Iter b) {
  return (b+a)/2;
}
