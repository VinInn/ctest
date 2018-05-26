#include<iostream>



int main() {
  constexpr int s=1024;
  int v[s];
  for (int i=0;i<s;++i) v[i]=2*i;
  std::cout << v[s-1] << std::endl;

  auto search = [&](int a, int j) { 
    if (a>=v[j]) {while(j<s && a>=v[++j]){} return j-1;}
    if (a<v[j]) {while(j>0 && a<v[--j]){} return j;}
    return j;
  };

  auto print = [&](int a) {
    std::cout << '\n' << a << ' ' << v[search(a,std::min(s-1,a/2))] << std::endl;
    std::cout << a << ' ' << v[search(a,0)] << std::endl;
    std::cout << a << ' ' << v[search(a,std::max(0,std::min(s-1,(a-8)/2)))] << std::endl;
    std::cout << a << ' ' << v[search(a,std::min(s-1,(a+8)/2))] << std::endl;
    std::cout << a << ' ' << v[search(a,s-1)] << std::endl;
  };

  print(0);
  print(v[s-1]);
  print(31);
  print(32);
  print(531);
  print(532);
  print(5000);

  return 0;

}
