//  cat /proc/meminfo | grep -i anon
//  ps -eo pid,command,rss,vsz | grep a.out
#include<iostream>
#include<vector>
#include<memory>
#include<malloc.h>

void stop(const char * m) {
  std::cout << m << std::endl;
  char c;
  std::cin >> c;
}



int main() {

  stop("start");

  constexpr size_t N = 1000*1000*1000;
/*
 {
    std::vector<int> v;
    v.reserve(N);
    stop("after reserve");
    v.resize(N);
    stop("after resize");
    v[0]=1;
    stop("after assign 0");
    for (int i=0; i<N; i+=10000) v[i]=1;
    stop("after assign many");

  }
*/

{
//    auto v = std::make_unique<int[]>(N);
    auto v = new int[N];
    stop("after create");
    v[0]=1;
    stop("after assign 0");
    for (int i=0; i<N; i+=10000) v[i]=1;
    stop("after assign many");

    delete [] v;

  }



  stop("stop");

  return 0;
}
