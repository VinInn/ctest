struct __attribute__((visibility("default"))) Foo {
    inline void foo() {};
    template <class C> inline void bar() {};
};

int main()
{
    Foo().foo();
    Foo().bar<Foo>();
}

