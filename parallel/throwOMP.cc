#include<omp.h>
#include<vector>

std::vector<std::vector<int> > v(100);

void bar(int i, std::vector<int> && a) {
  v[i]=a;
  if (a.size()==5) throw "error";
}

int main() {

#pragma omp parallel
  {
   // try {
      auto me = omp_get_thread_num();
      bar(me,std::vector<int>(me,2));
  //  } catch(...) {}

  }

  return 0;
}
