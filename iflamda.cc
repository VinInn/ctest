struct A {

int q;
bool sel;
  bool bar(int i, int j) const;
//  bool foo1(int i, int j) const;
//  bool foo2(int i, int j) const;
  bool foo3(int i, int j) const;
  bool foo4(int i, int j) const;

};


bool A:: bar(int i, int j) const {
  auto f = sel ? [](int a, int b) { return a<b;} :
                 [](int a, int b) { return b<a;};

  return f(i,j);
}


/*
bool A::foo1(int i, int j) const {
  auto f = sel ? [this](int a, int b) { return a<b;} :
                 [this](int a, int b) { return a<b+this->q;};

  return f(i,j);
}



bool A::foo2(int i, int j) const {
  int k = q;
  auto f = sel ? [k](int a, int b) { return a<b;} :
                 [k](int a, int b) { return a<b+k;};

  return f(i,j);
}

*/
bool A::foo3(int i, int j) const {
  auto f = sel ? [](int a, int b,int) { return a<b;} :
                 [](int a, int b, int k) { return a<b+k;};

  return f(i,j,q);
}


bool A::foo4(int i, int j) const {
  auto f1 = sel ? [](int a, int b,int) { return a<b;} :
                  [](int a, int b, int k) { return a<b+k;};

  auto f = [this,f1](int a, int b) { return f1(a,b,q);};
  return f(i,j);
}

