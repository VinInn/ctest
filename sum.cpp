#include<iostream>
#include<cmath>
#include<vector>

int main() {
  const int N=10000;
  std::vector<double> a(N), b(N,1.);
  a[0]=b[0]=0;

  for (int i=1;i!=N; i++) {
    a[i]=a[i-1]+exp(b[i]);
  }
  std::cout << a.back() << std::endl;

  for (int i=1;i!=N; i++) {
    a[i]=exp(b[i]);
  }
  for (int i=1;i!=N; i++) {
    a[i]+=a[i-1];
  }
  std::cout << a.back() << std::endl;


  return 0;
}
