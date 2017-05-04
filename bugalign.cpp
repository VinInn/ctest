#include <cstdint>
struct Base  {
    alignas(16) int64_t mAligned=0;
};

struct Derived : public virtual Base {
public:
    int64_t m1;
    int64_t m2{ 1234 };
    int64_t m3{ 2345 };

  __attribute__ ((noinline)) // or put ctor into different compilation unit
    Derived(){}
};

struct TestBug : public virtual Derived {};


int main() {
    Derived derived; // good
    TestBug testbug; // crashes because of using movaps on unaligned memory
}

