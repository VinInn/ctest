#include<iostream>
#include<cstdint>

inline
uint32_t  divu13(uint32_t n) {
uint32_t  q, r;
q = (n>>1) + (n>>4);
q = q + (q>>4) + (q>>5); q = q + (q>>12) + (q>>24); q = q >> 3;
r = n - q*13;
return q + ((r + 3) >> 4);
// return q + (r > 12);
}

// this is for the ROC n<512 (upgrade 1024)
inline
uint16_t  divu13(uint16_t n) {
uint16_t q, r;
q = (n>>1) + (n>>4);
q = q + (q>>4) + (q>>5); q = q >> 3;
r = n - q*13;
return q + ((r + 3) >> 4);
// return q + (r > 12);
}

// this is for the ROC n<512 (upgrade 1024)
inline
uint16_t  divu52(uint16_t n) {
uint16_t q, r;
n = n>>2; 
q = (n>>1) + (n>>4);
q = q + (q>>4) + (q>>5); q = q >> 3;
r = n - q*13;
return q + ((r + 3) >> 4);
// return q + (r > 12);
}



template<typename T>
void go() {

  T v[1024], u[1024];
  for (T i=0; i<1023; ++i) v[i]=i;

  for (int i=0; i<1023; ++i) u[i] = divu13(v[i]);

  for (int i=0; i<1023; ++i) 
    if(u[i]!=v[i]/13) std::cout << v[i] << ' ' << u[i] << std::endl;

  for (int i=0; i<1023; ++i) u[i] = divu52(v[i]);

  for (int i=0; i<1023; ++i)
    if(u[i]!=v[i]/52) std::cout << v[i] << ' ' << u[i] << ' ' << v[i]/52 << std::endl;


}

int main() {

   
  go<uint16_t>();
  go<uint32_t>();

  return 0;

}



