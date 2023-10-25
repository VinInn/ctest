#include<vector>


double dummy = 0;


#include<cstdio>
void go(int size) {

  printf("START\n");
  fflush(stdout); 

  std::vector<int> v;

  for (int i=0; i<size; ++i)
    v.push_back(i);

  dummy +=v[5];

}


int main() {

  go(100);

  return 0;
}
