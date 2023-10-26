#include<vector>


double dummy = 0;


#include<cstdio>

template<typename T, int mode>
void go(int size) {

  std::vector<int> v;
  if (mode==1) v.reserve(size);

  for (int i=0; i<size; ++i)
    v.push_back(T(i));

  dummy +=v[5];

}


int main() {

  printf("START\n");
  fflush(stdout);

  go<int,0>(100);
  go<int,1>(1000);
  go<double,0>(1000);

  return 0;
}
