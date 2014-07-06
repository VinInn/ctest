void swapit(char * fBufCur, int * ii, int n) {
for (int i = 0; i < n; i++) {
     ii[i] = __builtin_bswap32(*(int*)fBufCur);
     fBufCur += sizeof(int);
  }
}


/*
function Swap(const X: Int64): Int64;
const
SHUFIDX: array [0..1] of Int64 = ($0001020304050607, 0);
asm
MOVQ XMM0,[X]
PSHUFB XMM0,SHUFIDX
MOVQ [Result],XMM0
end;
*/

#include<cstring>
#include <emmintrin.h>
#include <tmmintrin.h>

const unsigned char maskByte[16] = {3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12};
union VI{
  VI(unsigned char const * i) {
    ::memcpy(b,i,16);
  }
  __m128i v;
  unsigned long long a[2];
  unsigned char b[16];

};

const VI maskV(maskByte);



inline __m128i swapBytes(__m128i x) {
  return  _mm_shuffle_epi8(x,maskV.v);
}


void swapit(__m128i *in, __m128i *out, int n) {
  // in and out must be 16 byte aligned
  for(int i = 0; i < n / sizeof(__m128i); i++ )
    {
      //__m128i	v = in[i];	 //load 16 bytes
      // v = _mm_or_si128( _mm_slli_epi16( v, 8 ), _mm_srli_epi16( v, 8 ) );	//swap it
      out[i] = swapBytes(in[i]);	 //store it out
    }
  
}
