#include<cstring>
inline __bf16 truncit(float float_val)
{
    __bf16 retval;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    memcpy(&retval, &float_val, sizeof retval);
#else
    memcpy(&retval, reinterpret_cast<char *>(&float_val) + sizeof float_val - sizeof retval, sizeof retval);
#endif
    return retval;
}
/*
  uint32_t lsb = (input >> 16) & 1;
  uint32_t rounding_bias = 0x7fff + lsb;
  input += rounding_bias;
  output = static_cast<uint16_t>(input >> 16);
*/


#include<cstdio>
#include<iostream>
#include<cmath>

void compare(float x) {
   __bf16 b1 = x;  // compiler
   __bf16 b2 = truncit(x); // truncate
   __bf16 b3 = truncit(x*1.001957f);  // round to nearest
  printf("%a :  gcc %a , trunc %a , round %a\n",x,float(b1),float(b2),float(b3));
}

int main() {
  compare(0x1.4f19fp+1);
  compare(0x1.fffbd4p+1);
}


void  scan() {

int k=0;
for (float x=0.5f; x<4.f; x+= 0.000004){ //  =std::nextafter(x,8.)) {
   __bf16 b1 = x;
   __bf16 b2 = truncit(x);
   __bf16 b3 = truncit(x*1.001957f);

   if (b1!=b2 || b1!=b3 || b2!=b3) printf("%a %a %a %a\n",x,float(b1),float(b2),float(b3));
   ++k;
}
}
