#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cmath>

/*  Quick and dirty, branchless, log implementations
    Author: Florent de Dinechin, Aric, ENS-Lyon 
    All right reserved

Warning + disclaimers:
- no special case handling (infinite/NaN inputs, even zero input, etc)
- no input subnormal handling, you'll get completely wrong results.
  This is the worst problem IMHO (leading to very rare but very bad bugs)
  However it is probable you can guarantee that your input numbers 
  are never subnormal, check that. Otherwise I'll fix it...
- output accuracy reported is only absolute. 
  Relative accuracy may be arbitrary bad around log(1), 
  especially for dirtylog0. dirtylogf is more or less OK.
- The larger/smaller the input x (i.e. away from 1), the better the accuracy.
- For the higher degree polynomials it is possible to win a few cycles 
  by parallelizing the evaluation of the polynomial (Estrin). 
  It doesn't make much sense if you want to make a vector function. 
- All this code is FMA-safe (and accelerated by FMA)
 
Feel free to distribute or insert in other programs etc, as long as this notice is attached.
    Comments, requests etc: Florent.de.Dinechin@ens-lyon.fr

Polynomials were obtained using Sollya scripts (in comments): 
please also keep these comments attached to the code of dirtylogf. 
*/



// #define DEGREE 3
// 2 gives you 7 bits, 8 gives you 24 bits 
// see the comments in the code for the accuracy you get from a given degree




typedef union {
  int32_t i32; /* Signed int */                
  float f;
} binary32;




////////////////////// Simplest, fastest version /////////////////////////
// Problem: log(1)<>0, so very poor relative accuracy on the output     //
inline float dirtylogf0(float x) {
	binary32 xx,m;
	xx.f = x;
	int E= (((xx.i32) >> 23) & 0xFF) -127;
	float Log2=0xb.17218p-4; // 0.693147182464599609375 is its exact binary32 representation
	m.i32 = (xx.i32 & 0x007FFFFF) | 0x3F800000;

	float y = m.f ;

#if (DEGREE==2)
 // 8-bit approx
	float p = -0x1.29332p0 + y * (0x1.6744ap0 + y * (-0x3.d311bp-4));
#elif (DEGREE==3) 
	// 11 bit approx
	float p=-0x1.7e26bcp0 + y * (0x2.1cd5dp0 + y * (-0xb.aa6dp-4 + y * 0x1.c14accp-4));
#elif (DEGREE==4) 
	// 14 bit approx
	float p= -0x1.bde4dcp0 + y * (0x2.d23768p0 + y * (-0x1.784cfcp0 + y * (0x7.279d2p-4 + y * (-0xe.7b676p-8))));
#else 
	float p=y-1; // very inaccurate 
#endif

		return (E*Log2+p);
}





///////// smarter, better behaved, but slightly slower version //////////////////
/// In doubt,  I would advise using this one  ////
inline float dirtylogf(float x) {
	binary32 xx,m;
	xx.f = x;

	// as many integer computations as possible, most are 1-cycle only, and lots of ILP.
	int E= (((xx.i32) >> 23) & 0xFF) -127; // extract exponent
	m.i32 = (xx.i32 & 0x007FFFFF) | 0x3F800000; // extract mantissa as an FP number

	int adjust = (xx.i32>>22)&1; // first bit of the mantissa, tells us if 1.m > 1.5
	m.i32 -= adjust << 23; // if so, divide 1.m by 2 (exact operation, no rounding)
	E += adjust;           // and update exponent so we still have x=2^E*y

	// now back to floating-point
	float y = m.f -1.0f; // Sterbenz-exact; cancels but we don't care about output relative error
	// all the computations so far were free of rounding errors...

	// the following is Sollya output

   // degree =  2   => absolute accuracy is  7 bits
#if ( DEGREE == 2 )
	float p =  y * ( float(0x1.0671c4p0) + y * ( float(-0x7.27744p-4) )) ;
#endif
   // degree =  3   => absolute accuracy is  10 bits
#if ( DEGREE == 3 )
   float p =  y * (0x1.013354p0 + y * (-0x8.33006p-4 + y * 0x4.0d16cp-4)) ;
#endif
   // degree =  4   => absolute accuracy is  13 bits
#if ( DEGREE == 4 )
   float p =  y * (0xf.ff5bap-4 + y * (-0x8.13e5ep-4 + y * (0x5.826ep-4 + y * (-0x2.e87fb8p-4)))) ;
#endif
   // degree =  5   => absolute accuracy is  16 bits
#if ( DEGREE == 5 )
   float p =  y * (0xf.ff652p-4 + y * (-0x8.0048ap-4 + y * (0x5.72782p-4 + y * (-0x4.20904p-4 + y * 0x2.1d7fd8p-4)))) ;
#endif
   // degree =  6   => absolute accuracy is  19 bits
#if ( DEGREE == 6 )
   float p =  y * (0xf.fff14p-4 + y * (-0x7.ff4bfp-4 + y * (0x5.582f6p-4 + y * (-0x4.1dcf2p-4 + y * (0x3.3863f8p-4 + y * (-0x1.9288d4p-4)))))) ;
#endif
   // degree =  7   => absolute accuracy is  21 bits
#if ( DEGREE == 7 )
   float p =  y * (0x1.000034p0 + y * (-0x7.ffe57p-4 + y * (0x5.5422ep-4 + y * (-0x4.037a6p-4 + y * (0x3.541c88p-4 + y * (-0x2.af842p-4 + y * 0x1.48b3d8p-4)))))) ;
#endif
   // degree =  8   => absolute accuracy is  24 bits
#if ( DEGREE == 8 )
   float p =  y * ( float(0x1.00000cp0) + y * (float(-0x8.0003p-4) + y * (float(0x5.55087p-4) + y * ( float(-0x3.fedcep-4) + y * (float(0x3.3a1dap-4) + y * (float(-0x2.cb55fp-4) + y * (float(0x2.38831p-4) + y * (float(-0xf.e87cap-8) )))))))) ;
#endif

   float Log2=0xb.17218p-4; // 0.693147182464599609375 
   return (float(E)*Log2+p);
}

/* The Sollya script that computes the polynomials above


f= log(1+y);
I=[-0.25;.5];
filename="/tmp/polynomials";
print("") > filename;
for deg from 2 to 8 do begin
  p = fpminimax(f, deg,[|0,23...|],I, floating, absolute); 
  display=decimal;
  acc=floor(-log2(sup(supnorm(p, f, I, absolute, 2^(-40)))));
  print( "   // degree = ", deg, 
         "  => absolute accuracy is ",  acc, "bits" ) >> filename;
  print("#if ( DEGREE ==", deg, ")") >> filename;
  display=hexadecimal;
  print("   float p = ", horner(p) , ";") >> filename;
  print("#endif") >> filename;
end;

*/


float vi[1024];
float vo[1024];
void foo() {
  for (int i=0; i!=1024;++i)
    vo[i] = dirtylogf(vi[i]);
}
     

/*
using namespace std;

int main() {
	float x;
	while (1+1==2) {
		cout << endl << "Enter x:";
    cin >> x;
    cout << endl << "dirtylogf0 =" << dirtylogf0(x);
    cout << endl << " dirtylogf =" << dirtylogf(x);
    cout << endl << "      logf =" << logf(x);
	}

}
*/
