#include <cmath>
#ifdef __MMX__
#include <mmintrin.h>
#endif

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __SSE3__
#include <pmmintrin.h>
#endif

void de() {
               #ifdef __SSE__
                _mm_setcsr (_mm_getcsr () | 0x8040);    // on Intel, treat denormals as zero for full speed
                #endif
}
typedef float T;

int main() { de(); T b; for (T x=1; x<50000000; x+=1) b=1./log10(x); return 0;}
