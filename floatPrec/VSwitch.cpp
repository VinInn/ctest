#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<iostream>

#include <x86intrin.h>

using float32x4_t = float __attribute__( ( vector_size( 16 ) ) );


using Float =  float;
using VFloat =  float32x4_t;



Float f0(Float );
Float f1(Float );
Float f2(Float );
Float f3(Float );
Float f4(Float );
Float f5(Float );



VFloat f0(VFloat );
VFloat f1(VFloat );
VFloat f2(VFloat );
VFloat f3(VFloat );
VFloat f4(VFloat );
VFloat f5(VFloat );


float lim[5];


// naive lookup switch
Float f(Float x) {
  if (x < lim[0] ) return f0(x); 
  if (x < lim[1] ) return f1(x); 
  if (x < lim[2] ) return f2(x); 
  if (x < lim[3] ) return f3(x); 
  if (x < lim[4] ) return f4(x); 
  return f5(x); 
}



VFloat f(VFloat x) {
  VFloat ret;
  // very naive lookup
  VFloat xl[6];
  xl[0] =  x < lim[0];
  for (int i=1;i!=5;++i)
    xl[i] =  x < lim[i] & x > lim[i-1] ;
   xl[5] =  x > lim[5];

// trivial switch
  if ( _mm_movemask_ps(xl[0]) != 0)
    ret = (x<lim[0]) ? f0(x) : ret;
  if ( _mm_movemask_ps(xl[1]) != 0)
    ret = (x<lim[1] & x > lim[0]) ? f1(x) : ret;
  if ( _mm_movemask_ps(xl[2]) != 0)
    ret = (x<lim[2] & x > lim[1]) ? f2(x) : ret;
  if ( _mm_movemask_ps(xl[3]) != 0)
    ret = (x<lim[3] & x > lim[2]) ? f3(x) : ret;
  if ( _mm_movemask_ps(xl[4]) != 0)
    ret = (x<lim[4] & x > lim[3]) ? f4(x) : ret;
  if ( _mm_movemask_ps(xl[5]) != 0)
    ret = (x>lim[4] ) ? f5(x) : ret;

  return ret;


}
