//compile and run with either
// c++ -std=c++23 testStacktrace.cpp -lstdc++exp -g -DINMAIN -DINLIB; ./a.out
// or
// c++ -std=c++23 testStacktrace.cpp -lstdc++exp -g -DINLIB -fpic -shared -o liba.so;c++ -std=c++23 testStacktrace.cpp -lstdc++exp -g -DINMAIN -L. -la -Wl,-rpath=.; ./a.out
//
#include <iostream>
#include <stacktrace>


#ifdef INLIB 
int nested_func2(int c)
{
    std::cout << std::stacktrace::current() << '\n';
    return c + 1;
}
int nested_func(int c)
{
    return nested_func2(c + 1);
}
#else
int nested_func(int c);
#endif
#ifdef INMAIN
int func(int b)
{
    return nested_func(b + 1);
}
 
int main()
{
    std::cout << func(777);
   return 0;
}
#endif
