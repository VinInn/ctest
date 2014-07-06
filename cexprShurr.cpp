#include <cstddef> 
#include <stdexcept> 

class str_const 
{ 
    const char* const p_; 
    const std::size_t  sz_; 
public: 
    template<std::size_t N> 
        constexpr str_const(const char(&a)[N]) : 
           p_(a), sz_(N-1) {} 

    constexpr char operator[](std::size_t n) 
    { 
        return n < sz_ ? p_[n] : throw std::out_of_range(""); 
    } 

    constexpr std::size_t size() { return sz_; } 
}; 

template <class T = std::size_t> 
constexpr 
inline 
T 
binary_const(str_const b, std::size_t n = 0, T x = 0) 
{ 
    return n == b.size() ? x : 
           b[n] == ',' ? binary_const<T>(b, n+1, x) : 
           b[n] == ' ' ? binary_const<T>(b, n+1, x) : 
           b[n] == '0' ? binary_const<T>(b, n+1, x*2) : 
           b[n] == '1' ? binary_const<T>(b, n+1, x*2+1) : 
            throw std::domain_error("Only '0', '1', ',', and ' ' may be used"); 
      //           static_assert(0=="Only '0', '1', ',', and ' ' may be used");  ///????
} 

int go() 
{ 
    constexpr str_const test("Hi Mom!"); 
    static_assert(test.size() == 7, ""); 
    static_assert(test[6] == '!', ""); 
    // constexpr unsigned i = binary_const("1111,0000");   // correct 
    constexpr unsigned i = binary_const("1111,0200");   // error
    return i; 
} 

int main() {
  return go();

}
