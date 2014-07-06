#if defined(NO_LIKELY)
#define likely(x) (x)
#define unlikely(x) (x)   
#elif defined(REVERSE_LIKELY)
#define unlikely(x) (__builtin_expect(x, true))
#define likely(x) (__builtin_expect(x, false))
#else
#define likely(x) (__builtin_expect(x, true))
#define unlikely(x) (__builtin_expect(x, false))
#endif
 


float __attribute__ ((aligned(16))) a[1024];
float __attribute__ ((aligned(16))) b[1024];


void foo() {
  for (int i=0; i!=1024; ++i) {
    b[i] = a[i]>0 ? a[i] : -a[i]+1;
  }
}


void bar() {
  for (int i=0; i!=1024; ++i) {
    b[i] = likely(a[i]>0) ? a[i] : -a[i]+1;
  }
}
