#include<memory>
#include<vector>
#include<functional>

#include<iostream>
#include<algorithm>


struct A {
  A(int ii=-1) : i(ii){}
  int i;
};

typedef std::shared_ptr<A> Aptr;

struct Aless {
  bool operator()(Aptr const & lh, Aptr const &rh) {
    return lh->i < rh->i;
  }
};

int main() {
  
  std::vector<Aptr> v;
  v.push_back(Aptr(new A(2)));
  v.push_back(Aptr(new A(0)));
  v.push_back(Aptr(new A(1)));
  v.push_back(v[1]);
  
  for (auto p=v.begin(); p!=v.end(); ++p)
    std::cout <<  (*p)->i << " ";
  std::cout << std::endl;


  std::stable_sort(v.begin(),v.end(),Aless());

  for (auto p=v.begin(); p!=v.end(); ++p)
    std::cout <<  (*p)->i << " ";
  std::cout << std::endl;

}
