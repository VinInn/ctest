#include<bitset>

using B8=std::bitset<8>;

unsigned char b;

void set(int i) {
  B8 a(b); a.set(i);
  b=a.to_ulong();
}

void bset(int i) {
   b |= 1<<i;
}
