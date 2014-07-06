// -Wold-style-cast  -Woverloaded-virtual -Wctor-dtor-privacy  -Wabi -Weffc++

void f();

enum Error {WARNING, SEVERE, FATAL};

class A {
  
  virtual void bha()=0;


  inline int echo (int q) {a=q;}


  typedef int Int;
  
  Int a;

};

// inline A::Int A::echo (A::Int q) { return q;}


class B : public A {
public:
  B(){}
  
  B(B& a) {}

  void operator=(const B&a) {}

  void bha() {
    f();
  }

  int hi() {

  }

  int a;

};

class Q {
public:
  Q() : a(0){}

  void operator ++() { 
    ++a; // return *this; 
  }

  void operator ++(int) { 
    Q q(*this); ++a; // return q;
  }

  void operator +=(Q& q) {
    a+=q.a;
  }

private:
  int a;
};

class C : public B {
public:
  C() : q() {}
  void bha() const;
  void operator=(C c);
private:
  Q q;
};

int main() {

  float b=3.3;

  float c = int(b);

  A * q = new B;

  B * aa = (B*)(q);

  if (c) {
    B a;
  }
  

}
