#include "BlockAllocator.h" 
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <algorithm>

struct B {
  virtual ~B(){}
  virtual B* clone() const =0;
  int i;

};




struct A1 : public B
#ifdef BALLOC
, public BlockAllocated<A1,1000> 
#endif
{
  virtual A1* clone() const {
    return new A1(*this);
  }

  double a;
  char c;
  bool b;

};

struct A2 : public B
#ifdef BALLOC
, public BlockAllocated<A2,1000> 
#endif
{
  virtual A2* clone() const {
    return new A2(*this);
  }
  
  float a;
  char c[2];
  long long  l;

};

/*
BlockAllocated<A1>::s_allocator =  BlockAllocated<A1>::allocator(10);
BlockAllocated<A2>::s_allocator =  BlockAllocated<A2>::allocator(100);
*/


void dump(std::string const & mess="") {
#ifdef DEBUG
  std::cout << mess << std::endl;
  BlockAllocator::Stat sa1 = BlockAllocated<A1,1000>::stat();
  BlockAllocator::Stat sa2 = BlockAllocated<A2,1000>::stat();
  std::cout << "A1 " << sa1.blockSize
	    << " " << sa1.currentOccupancy
	    << " " << sa1.currentAvailable
	    << " " << sa1.nBlocks 
	    << std::endl;
  std::cout << "A2 " << sa2.blockSize
	    << " " << sa2.currentOccupancy
	    << " " << sa2.currentAvailable
	    << " " << sa2.nBlocks 
	    << std::endl;
#endif
}

typedef boost::shared_ptr<B> BP;

void gen(BP & bp) {
  static bool flip=false;
  if (flip) 
    bp.reset(new A1);
  else
    bp.reset(new A2);
  flip = !flip;
}

int main() {
  
  dump();


  {
    BP b1(new A1);
    BP b2(new A2);
    dump();
    {
      BP bb1(b1->clone());
      BP bb2(b2->clone());
      dump();
    }
    dump("after clone destr");

    BP b11(new A1);
    BP b22(new A2);
    dump();
    b1.reset();
    b2.reset();
    dump("after first destr");
  }

  dump();
  { 
    std::vector<BP> v(233);
    for_each(v.begin(),v.end(),&gen);
    dump("after 233 alloc");
    v.resize(123);
    dump("after 110 distr");
  }
  dump();

  for (int i=0;i<100;i++){
    std::vector<BP> v(2432);
    for_each(v.begin(),v.end(),&gen);
    std::vector<BP> v1(3213);
    for_each(v1.begin(),v1.end(),&gen);
    {
      std::vector<BP> d; d.swap(v);
    }
    // alloc disalloc
    std::vector<BP> vs(514);
    for_each(vs.begin(),vs.end(),&gen);
    for_each(vs.begin(),vs.end(),&gen);
  }

  return 0;
}

