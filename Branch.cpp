// c++-52 -std=c++14 -O3 -Wall Branch.cpp  -fopt-info-vec
#include<algorithm>


inline float branch(float x, float y, float z) {
   float ret=0;
   if (x<0 && y<0 && z<0)  ret=x;
   else if(y>0 || z>2.f) ret+=y;
   else if(x>y && z<y) ret-=z;
   return ret;
}


inline float branchless(float x, float y, float z) {
   return
      (x<0) & (y<0) & (z<0) ?  x : 
     (  (y>0) | (z>2.f) ? y :
        ( (x>y) & (z<y) ? z : 0 )
     );
}

inline float branchless2(float x, float y, float z) {
   auto r1 = (x>y) & (z<y) ? z : 0;
   auto r2 = (y>0) | (z>2.f) ? y : r1;
   return   (x<0) & (y<0) & (z<0) ?  x : r2;
}


void init(float * x, int N, float y) {
   for ( int i = 0; i < N; ++i ) x[i]=y+i-float(N/2); // try to remove -float(N/2) 
   std::random_shuffle(x,x+N);                       // and or this
}


float * alloc(int N) {
  return new float[N];

}


#include<iostream>
int main() {

   int N = 1000;

   int size = N*N;
   float * a = alloc(size);
   float * b = alloc(size);
   float * c = alloc(size);
   float * r = alloc(size);

  init(c,size,0.f);
  init(a,size,1.3458f);
  init(b,size,2.467f);


  double s=0;
  for (int i=0; i<1000; ++i) {
    for(int j=0;j<size; ++j) r[j]=branchless2(a[j],b[j],c[j]);
    s+=r[i];
  }

  std::cout<<s<<std::endl;
  return s;

}

