/*
Inspired from:
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1995, 2000 by Stephen L. Moshier
*/


#include <iostream>
#include "math.h"

namespace vdt{

double p1evl( double x, double* coef, int N ){
double ans;
double *p;
int i;

p = coef;
ans = x + *p++;
i = N-1;

do
        ans = ans * x  + *p++;
while( --i );

return( ans );
}

double polevl( double x,double* coef,int N )
{
double ans;
int i;
double *p;

p = coef;
ans = *p++;
i = N;

do
    ans = ans * x  +  *p++;
while( --i );

return( ans );
}


double ldexp( double x,int pw2 ){
// Dumb thing, to be replaced with the shift
return x * pow(2,pw2);

}


/* Coefficients for log(1+x) = x - x**2/2 + x**3 P(x)/Q(x)
 * 1/sqrt(2) <= x < sqrt(2)
 */
static double P[6] = {
 1.01875663804580931796E-4,
 4.97494994976747001425E-1,
 4.70579119878881725854E0,
 1.44989225341610930846E1,
 1.79368678507819816313E1,
 7.70838733755885391666E0,
};
static double Q[5] = {
//  1.00000000000000000000E0, 
 1.12873587189167450590E1,
 4.52279145837532221105E1,
 8.29875266912776603211E1,
 7.11544750618563894466E1,
 2.31251620126765340583E1,
};

/* Coefficients for log(x) = z + z**3 P(z)/Q(z),
 * where z = 2(x-1)/(x+1)
 * 1/sqrt(2) <= x < sqrt(2)
 */

static double R[3] = {
-7.89580278884799154124E-1,
 1.63866645699558079767E1,
-6.41409952958715622951E1,
};
static double S[3] = {
// 1.00000000000000000000E0,
-3.56722798256324312549E1,
 3.12093766372244180303E2,
-7.69691943550460008604E2,
};

#define SQRTH 0.70710678118654752440

double log(double x){
    int e;
    double y, z;

    /* separate mantissa from exponent */
    /* Note, frexp is used so that denormal numbers
    * will be handled properly.
    */
    x = frexp( x, &e );

    /* logarithm using log(x) = z + z**3 P(z)/Q(z),
    * where z = 2(x-1)/x+1)
    */


#ifndef NOSMALL
    if( (e > 2) || (e < -2) ){
        If( x < SQRTH ){ // 2( 2x-1 )/( 2x+1 )
            e -= 1;
            z = x - 0.5;
            y = 0.5 * z + 0.5;
            }	
        else{ //  2 (x-1)/(x+1)  
            z = x - 0.5;
            z -= 0.5;
            y = 0.5 * x  + 0.5;
            }
        x = z / y;

        // rational form
        z = x*x;
        z = x * ( z * polevl( z, R, 2 ) / p1evl( z, S, 3 ) );
        y = e;
        z = z - y * 2.121944400546905827679e-4;
        z = z + x;
        z = z + e * 0.693359375;
        return( z );
    
        }
#endif    
  
    
    /* logarithm using log(1+x) = x - .5x**2 + x**3 P(x)/Q(x) */

    if( x < SQRTH ){
	    e -= 1;
	    x = ldexp( x, 1 ) - 1.0; /*  2x - 1  */
	    }	
    else
        x = x - 1.0;

    /* rational form */
    z = x*x;
    y = x * ( z * polevl( x, P, 5 ) / p1evl( x, Q, 5 ) );
    if( e )
        y = y - e * 2.121944400546905827679e-4;
 
    // LD exp to be done with the shift
    y = y - ldexp( z, -1 );   /*  y - 0.5 * z  */
    z = x + y;
    if( e )
        z = z + e * 0.693359375;

    return( z );
    }
}// end of vdt namespace


#include <stdlib.h>
#include <iomanip>

int main(int argc, char** argv){

    double x=atof(argv[1]);
    std::cout << std::setprecision(20);
    std::cout << "Log of exp(e, " << x << ")  is vdt: " << vdt::log(exp(x)) << " std: "<< log(exp(x)) << std::endl;
    std::cout << "Log of " << x << " is vdt: " << vdt::log(x) << " std: "<< log(x) << std::endl;



}

