

class A {
public:

  int i;

  static __thread int j;

  static int k() {
    static __thread int l;
    return l;
  }

};

int A::j;

__thread int k;

int main() {



  A a;

  a.i = 3;

  A::j = a.i;

  k = A::j;

  return 0;

}
