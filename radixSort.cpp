#include<cstdint>
#include<cassert>

void radixSort(int16_t const * a, uint32_t * ind, uint32_t size) {
  constexpr int d = 8, w = 16;
  constexpr int sb = 1<<d;
  auto j = ind;
  uint32_t ind2[size];
  auto k = ind2;
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
    // swap
    auto t=j;j=k;k=t;
    // for (int i = 0; i < size; i++) j[i]=k[i];
  }
  // assume w/d is even
  assert(j==ind);

   // now move negative first...
  // find first negative
  int firstNeg=0;
  for (int i = 0; i < size-1; i++) {
    //if ( (int(a[ind[i]])*int(a[ind[i+1]])) <0 ) firstNeg=i+1;
    if ( (a[ind[i]]^a[ind[i+1]]) < 0 ) firstNeg=i+1;
  }
  assert(firstNeg>0);

  auto ii=0;
  for (auto i=firstNeg; i<size; i++) ind2[ii++] = ind[i];
  assert(ii = size-firstNeg);
  for (auto i=0;i<firstNeg;i++)  ind2[ii++] = ind[i];
  assert(ii=size);
  
  for (int i = 0; i < size; i++) ind[i]=ind2[i];
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
  std::cout << v[ind[5999]] << ' ' << v[ind[6000]] << ' ' << v[ind[6001]] << std::endl;
  for (int i = 1; i < N; i++) {
    if (v[ind[i]]<v[ind[i-1]])
      std::cout << "not ordered at " << ind[i] << " : "
		<< v[ind[i]] <<' '<< v[ind[i-1]] << std::endl;
  }
  return 0;
}
