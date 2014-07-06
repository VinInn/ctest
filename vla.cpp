#include<algorithm>
#include<cassert>
namespace mine {
  /*
  template<class _Container>
  inline auto
  end(_Container& __cont) -> decltype(__cont.end())
  { return __cont.end(); }
  template<class _Container>
  inline auto
  end(const _Container& __cont) -> decltype(__cont.end())
  { return __cont.end(); }
  */  
  template<typename T> 
  inline T * 
  end(T t[]) { return t+sizeof(t)/sizeof(T);}

  //  template<typename T, int N> 
  //inline T * 
  //end(T (&t)[N]) { return t+sizeof(t)/sizeof(T);}
}

#define END(VLA) VLA+sizeof(VLA)/sizeof(*VLA)

// #ifdef VLA
int foo(int N) {
  int v[N];
  auto a = mine::end(v);
  return *(a-1);
}


int foo2(int N) {
  int v[N];
  for ( auto a : v)
    if (a) return a;
  return 0;
}


// #endif

int bar() {
  int v[10];
  auto a = mine::end(v);
  return *(a-1);
}

int bar2() {
  int v[10];
  for ( auto a : v)
    if (a) return a;
}

std::vector<int> reverse(std::vector<int> & v) {
  int N=v.size();
  std::vector<int> r; r.reserve(N);
  int tmp[N];
  int i=0;
  for ( int k : v) tmp[i++]=k;
  assert(i==N);
  while(i!=0) r.push_back(tmp[--i]);
  return r;
}


#include <cstdio>
int main(int n, const char**) {
  double d[n];
  char c[n];
  
  printf("%d %d\n", sizeof(d),sizeof(c));

  printf("%d %d\n", mine::end(d)-d,mine::end(c)-(c));
  printf("%d %d\n", END(d)-d,END(c)-(c));


  double cd[10];
  char cc[10];
 
  printf("\n%d %d\n", sizeof(cd),sizeof(cc));

  printf("%d %d\n", mine::end(cd)-cd,mine::end(cc)-(cc));
  printf("%d %d\n", END(cd)-cd,END(cc)-(cc));


  std::vector<int> vv= {0,1,2,3,4,5,6,7,8,9};
  auto v = reverse(vv);
  for ( int k : v) printf("%d ",k); printf("\n");

}
