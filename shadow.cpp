#include<iostream>


class Base {
public:

void count(int i) {
  std::cout << "Base " << i << std::endl;
  k[i]++;  
}

  unsigned int k[100];;

};


template<typename T>
class H : public Base {
public:

void count(T t) {
  std::cout << "H " << t << std::endl;
  int i = t;
  i = std::min(99,std::max(0,i));
  k[i]++;
}

};



int main() {

H<float> hf;
hf.count(3.14);

H<short> hs;
hs.count(3);

H<unsigned int> hu;
hu.count(3);

H<int> hi;
hi.count(3);


  return 0;
}
