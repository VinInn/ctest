#include<iostream>
#include<cassert>


void check(int n) {

  int a[n];
  bool b[n];
  for (int i=0; i<n; i++){
    a[i]=1; b[i]=false;
  }
  for (int i=0; i<n; i++){
   assert(a[i]==1);
   assert(!b[i]);
 }

}


int main(int v, char**) {

  if (v>2) {
    check(10);
  }
  check(20);

  return 0;
}
