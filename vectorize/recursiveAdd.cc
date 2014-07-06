#include<algorithm>

template<typename T, int N>
struct Add {

  T operator()(T const * const * v, T const * c, int i ) const { return (*c)*(*v)[i] + add(v+1,c+1,i); }
  T operator()(std::initializer_list<T> il) const { return *il.begin()+add(std::begin(il)+1);}
  Add<T,N-1> add;
  
};

template<typename T>
struct Add<T,2> {
  
  T operator()(T x, T y) const { return x+y;}
  T operator()(T const * const * v, T const * c, int i ) const { return (*c)*(*v)[i] + (*(c+1))*(*(v+1))[i]; }
  T operator()(std::initializer_list<T> il) const { return *il.begin() + *(std::begin(il)+1);}
};




float x1[1024];
float x2[1024];
float x3[1024];
float x4[1024];
float z[1024];
float c[4];

void add() {

  float * x[4]={x1,x2,x3,x4};

  Add<float,4> add;
  for (int i=0; i!=1024; ++i) {
    z[i] = add(x,c,i);
  }

}



#include<cassert>
#include<iostream>
int main() {

  for (int i=0; i!=1024; ++i) {
    x1[i]=i;
    x2[i]=10*i;
    x3[i]=0.5*i;
    x4[i]=-40*i;
  }

  c[0]=1; c[1]=2; c[2]=-2; c[3]=0.5;
  add();
  
  for (int i=0; i!=1024; ++i) {
    if (std::abs(z[i])>1.e-5) std::cout << z[i] << std::endl;;
  }

  return 0;
}
