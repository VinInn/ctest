#include<map>
#include<vector>

int main() {
  typedef std::vector<int> M;
  // typedef std::map<int,int> M;
  typedef M::const_reverse_iterator const_iterator;
  // typedef M::reverse_iterator const_iterator;

  const M m;
  const_iterator p;

  if (p==m.rend()) p=m.rbegin();

  return 0;


}
