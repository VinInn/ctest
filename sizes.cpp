#include<iostream>

struct xyz {
  float x,y,z;
};

struct xyzw {
  float x,y,z,w;
} __attribute__((aligned(16))); 


struct Z {
  // double d;
  void * p;
  // int w; int q;
  int a; int b;
};


struct A {
  int w; int q;
  int a;
};

struct A1 : public A {

  int b;

};


struct A2 {
  A a;
  int b;

};

struct C {
  void * p2;
  int a;
};

struct C1 : public C {

  int b;

};


struct C2 {
  C a;
  int b;

};


struct B {
  virtual ~B(){}
  //  virtual void q()=0;
  // void * p;
  int a;
};

struct B1 : public B {
  //  virtual void q(){}

  int b;

};

struct B2 {
  B a;
  int b;

};


struct H {
  virtual ~H(){}
  //  virtual void q()=0;
  unsigned int id :30;
  unsigned int st :2;
};

struct H1 : public H {
  xyz p;
};


struct H2 : public H {
  xyz p;
  xyzw l;
};

struct H3 : public H {
  xyz p;
  xyz l;
};


#ifndef offsetof
#define offsetof(type, member)  __builtin_offsetof (type, member)
#endif

struct MxAling {} __attribute__((__aligned__));

int main() {

   std::cout << sizeof(MxAling) << " " << alignof(MxAling) << std::endl;

  std::cout << sizeof(Z) << " " << alignof(Z) << std::endl;
  std::cout << sizeof(xyz) << " " << alignof(xyz) << std::endl;
  std::cout << sizeof(xyzw) << " " << alignof(xyzw) << std::endl;
  std::cout << sizeof(A) << " " << alignof(A) << std::endl;
  std::cout << sizeof(A1) << " " << alignof(A1) << std::endl;
  std::cout << sizeof(A2) << " " << alignof(A2) << std::endl;
  std::cout << sizeof(C) << " " << alignof(C) << std::endl;
  std::cout << sizeof(C1) << " " << alignof(C1) << std::endl;
  std::cout << sizeof(C2) << " " << alignof(C2) << std::endl;
  std::cout << sizeof(B) << " " << alignof(B) << std::endl;
  std::cout << sizeof(B1) << " " << alignof(B1) << std::endl;
  std::cout << sizeof(B2) << " " << alignof(B2) << std::endl;
  std::cout << sizeof(H) << " " << alignof(H) << std::endl;
  std::cout << sizeof(H1) << " " << alignof(H1) << std::endl;
  std::cout << sizeof(H2) << " " << alignof(H2) << std::endl;
  std::cout << sizeof(H3) << " " << alignof(H3) << std::endl;

  std::cout << std::endl;
  std::cout << offsetof(C,a) << std::endl;
  
  
};
