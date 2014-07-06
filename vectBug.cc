const int arraySize=512;

struct Bar {
  
  int __attribute__ ((aligned(16))) c[arraySize];
  int last;

  Bar() : last(0) { refresh();}

  void refresh();

  void loop0(int N, float * f) {
    int k=0;
    int lead = arraySize-last;
    if (N<=lead) {
      for (int i=0; i!=N; ++i) f[k++] = c[last++];
      return;
    }
    
    for (int i=last; i!=arraySize; ++i)  f[k++] = c[i];
    int outLoop = (N-lead)/arraySize;
    last = N -lead -  outLoop*arraySize;
    for (int j=0; j!=outLoop; ++j)  {
      refresh();
      for (int i=0; i!=arraySize; ++i) f[k++] = c[i];
    }
    refresh();
    for (int i=0; i!=last; ++i) f[k++] = c[i];
  }

  template<typename F>
  void loop(int N, F f) {
    int lead = arraySize-last;
    if (N<=lead) {
      for (int i=0; i!=N; ++i) f(c[last+i]);
      last +=N;
      return;
    }
    
    for (int i=last; i!=arraySize; ++i)  f(c[i]);
    int outLoop = (N-lead)/arraySize;
    last = N -lead -  outLoop*arraySize;
    for (int j=0; j!=outLoop; ++j)  {
      refresh();
      for (int i=0; i!=arraySize; ++i) f(c[i]);
    }
    refresh();
    for (int i=0; i!=last; ++i) f(c[i]);
  }

};


float __attribute__ ((aligned(16))) z[4096];
void refresh();
int j=0;


void fun(float const *, float const *, int); 


template<typename F>
inline void loop(int N, F f) {
  if (j+N>4096) {
    j=0;
    refresh();
  }
  for (int i=0; i!=N; ++i) f(z[j++]);
}

void foo(int N) {
  float __attribute__ ((aligned(16))) x[N];
  float __attribute__ ((aligned(16))) y[N];
  int k=0;
  auto xs = [&x, &k](float r) { x[k++]= 1.5f*r;};
  auto ys = [&y, &k](float r) { y[k++]= r+1.f;};


  k=0;
  loop(N,xs);
  // for (int i=0; i!=N; ++i) xs(z[j++]);
    // x[k++] = z[j++];

  k=0;
  loop(N,ys);

  //  for (int i=0; i!=N; ++i) ys(z[j++]);
  //    y[k++] = z[j++];

  fun(x,y,N);
}


void load(int N) {

float __attribute__ ((aligned(16))) a[N];
#ifndef FIXED
float __attribute__ ((aligned(16))) b[N];
#else
float __attribute__ ((aligned(16))) b[1024];
#endif

  static Bar bar;


  bar.loop0(N,a);
  bar.loop0(N,b);
  fun(a,b,N);

  

  int k=0;
  auto as = [&a, &k](float r) { a[k++]= 1.5f*r;};
  auto bs = [&b, &k](float r) { b[k++]= r+1.f;};

  k=0;
  bar.loop(N,as);
  k=0;
  bar.loop(N,bs);
  

  fun(a,b,N);

}

