#include<memory>
#include<vector>
#include<functional>

#include<iostream>

void print(char const * hi, int i) {
  std::cout << hi << " " << i << std::endl;
}

struct D {
  void operator()(void *){}
};

int main() {


  std::shared_ptr<int> p{new int(3)};
  std::weak_ptr<int> w{p};

  std::unique_ptr<int> u{new int(5)};

  std::vector<std::unique_ptr<int> > vu;

  auto fakeDel = std::bind(print, "hello world",std::placeholders::_1);

  fakeDel(*p);
  fakeDel(*u);

  {
    std::shared_ptr<void> guard((void*)(0),std::bind(print, "hello world", *p));
    

  }


  return 0;
}
