namespace a {
  namespace b {

    template<typename T> struct tr {
      typedef T value;
      void null(){}
    };
  }
}

namespace a {
  namespace b {
    template<> struct tr<long> {typedef int value; static short k;};
  }
}
short a::b::tr<long>::k;

int main() {

  return sizeof(a::b::tr<long>::value);

}


