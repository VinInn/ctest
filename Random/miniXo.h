#include <cstdint>
 
 template<typename T>
  static constexpr T rotl(const T x, int k) {
    return (x << k) | (x >> (64 - k));
  }

unsigned long long  a[1024];
unsigned long long  b[1024];
void rot() {
   for (int i=0; i<1024; ++i) b[i] = rotl(a[i],25);
}


struct SOA {
  constexpr uint64_t * operator[](int i) { return v[i];}
  constexpr uint64_t const * operator[](int i) const { return v[i];}
  uint64_t* v[4];
};

constexpr void advance(SOA & s, int i) {

    const auto t = s[1][i] << 17;

    s[2][i] ^= s[0][i];
    s[3][i] ^= s[1][i];
    s[1][i] ^= s[2][i];
    s[0][i] ^= s[3][i];

    s[2][i] ^= t;

    s[3][i] = rotl(s[3][i], 45);
}

constexpr auto nextPP(SOA & s, int i) {
    const auto result = rotl(s[0][i] + s[3][i], 23) + s[0][i];
    advance(s,i);
    return result;
}



void gen(SOA s, uint64_t * a, int n) {
    auto ni = n/32; 
    for (int i = 0; i < ni; i++) {
      int j=0;
      for (int k=0; k<32; ++k)
        a[j++] = nextPP(s,k);
    }    
}

