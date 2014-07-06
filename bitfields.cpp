#include<vector>
#include<iostream>

struct Fields {
  unsigned int PFN : 22;
  int : 2;
  unsigned int CCA :3;
  bool nonreachable : 1;
  bool dirty : 1;
  bool valid : 1;
  bool global : 1;
  
};

union  Store {
  unsigned int store;
  Fields w;
};



struct Digi {
  
  Fields w() const { return (Fields const&)(m_w);} 
  Fields & w() { return (Fields&)(m_w);} 

  unsigned int m_w;

};


struct Data : public std::vector<Digi> {
  virtual ~Data(){}
};



int main() {


  Digi d;
  d.w().CCA = 4;
  d.w().valid=false;

  std::cout << d.w().CCA << " " <<  d.w().valid << std::endl;

  return 0;
}
