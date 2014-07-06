#include<typeinfo>


class B {
public:
  virtual bool isValid() const { return true;}

};

class I : public B {
public:
  virtual bool isValid() const { return false;}

};

class G : public B {
public:

};

class DG : public G {
public:

};


bool isGood_dc(B const * b) {
  return dynamic_cast<I const*>(b)==0;
}

bool isGood_ti(B const * b) {
  return typeid(I)!=typeid(*b);
}

bool isGood_vi(B const * b) {
  return (*b).isValid();
}


int main() {

  B * b = new DG();

  isGood_dc(b);
  isGood_ti(b);
  isGood_vi(b);


}
