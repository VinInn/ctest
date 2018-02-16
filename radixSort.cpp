#include<cstdint>

void radixSort(int16_t const * a, uint32_t * ind, uint32_t size) {
  constexpr int d = 8, w = 16;
  constexpr int sb = 1<<d;
  uint32_t * j = ind;
  uint32_t k[size];
  for (int p = 0; p < w/d; p++) {
    int c[sb];
    for (int i = 0; i < sb; i++) c[i]=0;
    // the next three for loops implement counting-sort
    for (int i = 0; i < size; i++)
      c[(a[j[i]] >> d*p)&(sb-1)]++;
    
    for (int i = 1; i < sb; i++)
      c[i] += c[i-1];
    
    for (int i = size-1; i >= 0; i--)
      k[--c[(a[j[i]] >> d*p)&(sb-1)]] = j[i];
    
    for (int i = 0; i < size; i++)
      j[i]=k[i];
  }
}


#include<algorithm>
#include<cassert>
#include<iostream>
int main() {

  constexpr int N=12000;
  int16_t v[N];
  uint32_t ind[N];

  for (int i = 0; i < N; i++) {
    ind[i]=i; v[i]=i%32768; if(i%2) v[i]=-v[i];
  }

  radixSort(v,ind,N);

  std::cout << v[ind[10]] << ' ' << v[ind[11000]] << std::endl;
  std::cout << v[ind[5999]] << ' ' << v[ind[6000]] << std::endl;
  for (int i = 1; i < N; i++) {
    assert(v[ind[i]]>=v[ind[i-1]]);
  }
  return 0;
}
