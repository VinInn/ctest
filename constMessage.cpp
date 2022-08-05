
#include <cstdint>


    // FNV-1a 32bit hashing algorithm.
    constexpr uint32_t fnv1a_32_impl(char const* s, int count)
    {
        return ((count ? fnv1a_32_impl(s, count - 1) : 2166136261u) ^ s[count]) * 16777619u;
    }
    constexpr uint32_t fnv1a_32(char const* s) {
        return fnv1a_32_impl(s,sizeof(s)-2);
    }


struct Message {

  constexpr Message(const char * s) : m(s), h(fnv1a_32(s)){}
  const char * const m;
  const  uint32_t h;
};

constexpr Message err[2] = {
    Message("first message"),
    Message("second message")
   };


uint32_t run[100]; 

void fun(int l) {

   if (l==0) run[0] = err[0].h;   

}


#include <iostream>

int main(int argc, char** argv) {

   constexpr const char * m1 = "first message";
   constexpr uint32_t h1 = fnv1a_32(m1);
   std::cout << m1 << ' ' << sizeof("first message") << ' ' << h1 << std::endl;

//   std::cout << mm[0] << ' ' << hh[0] << std::endl;

   fun(argc-1);
   std::cout << run[0] << std::endl;


   return 0;

};
