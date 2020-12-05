#include "FlexiStorage.h"



#include<cassert>

using namespace cms::cuda;

int main() {


  FlexiStorage<int,1024> a;

  assert(a.capacity()==1024);

  FlexiStorage<int,-1> v;

  v.init(a.data(),20);

  assert(v.capacity()==20);

  assert(v.data()==a.data());

  return 0;


};
