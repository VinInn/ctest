
#include<iostream>

template<char C>
struct V {
  static void hello() { std::cout << "new" << std::endl;}
};


template<>
struct V<'5'> {
  static void hello() { std::cout << "old" << std::endl;}
};


constexpr char major(const char * c) { return c[6];}
#define xstr(s) str(s)
#define str(s) #s



#include<iostream>
int main() {

V<major(xstr(PROJECT_VERSION))>::hello();

#if ( major(xstr(PROJECT_VERSION)) > '5' )
std::cout << "new" << std::endl;
#else
std::cout << "old" << std::endl;
#endif

return 0;


}
