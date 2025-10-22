#include<cstring>

inline __bf16 conv(float float_val)
{
    __bf16 retval;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    memcpy(&retval, &float_val, sizeof retval);
#else
    memcpy(&retval, reinterpret_cast<char *>(&float_val) + sizeof float_val - sizeof retval, sizeof retval);
#endif
    return retval;
}



#include<cstdio>
#include<iostream>
#include<cmath>

int main() {

int k=0;
for (float x=0.5f; x<4.f; x+= 0.000004){ //  =std::nextafter(x,8.)) {
   __bf16 b1 = x;
   __bf16 b2 = conv(x);
   __bf16 b3 = conv(x*1.001957f);

   if (b1!=b2 || b1!=b3 || b2!=b3) printf("%a %a %a %a\n",x,b1,b2,b3);
   ++k;
}

  return 0;

}
