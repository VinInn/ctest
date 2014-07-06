struct A {
  static const int N=32;

  typedef bool FilterType(unsigned int);
  typedef void (A::*MemFuncType)() const;

  int count(FilterType F) const {
    int c=0;
    for (unsigned int i=0; i!=N; ++i) 
      if (F(a[i])) c++;
    return c;
    }
    
    static bool f1(unsigned int z) { return 0==z%2; }
    static bool f2(unsigned int z) { return z>0; }

    int c1() const {
      return count(f1);
    }
    int c2() const {
      return count(f2);
    }

  unsigned int a[N];
  
};
