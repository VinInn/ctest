#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <limits>


typedef union {
    double d;
    long long ll;
} ieee754;

static const double SQRTH =0.70710678118654752440;

double vdt_frexp(double x, int* exponentp){
  
    double d = x;
    int exponent = *exponentp;
    ieee754 u;
    u.d=d;

    // Note that the shift is sign-extended, hence the test against -1 not 1
    int negative = 1 - 2 *(int)(u.ll < 0);
    exponent = (int) ((u.ll >> 52) & 0x7ffL);
    long mantissa = u.ll & 0xfffffffffffffL;

    // Subnormal numbers; exponent is effectively one higher,
    // but there's no extra normalisation bit in the mantissa
    if (exponent==0)
        exponent++;
    // Normal numbers; leave exponent as it is but add extra
    // bit to the front of the mantissa
    else
        mantissa = mantissa | (1LL<<52);

    // Bias the exponent. It's actually biased by 1023, but we're
    // treating the mantissa as m.0 rather than 0.m, so we need
    // to subtract another 52 from it.
    exponent -= 1075;

    /* Normalize */
   while((mantissa & 1) == 0) 
   {    /*  i.e., Mantissa is even */
       mantissa >>= 1;
       exponent++;
   }   

   std::cout << "Before second normalisation\n";
   std::cout << "mantissa = " << mantissa << " exp = " << exponent << std::endl;
   std::cout <<  d << " * 2^" << exponent << " "<< mantissa*pow(2,exponent)<< std::endl;   
   
   d=mantissa;    
   
   while(d >= 1) 
   {    /*  i.e., Mantissa is > 1 */
       d/=2;
       exponent++;
   }   
  *exponentp=exponent;
  
    std::cout << "mantissa = " << d << " exp = " << exponent << std::endl;
    std::cout <<  d << " * 2^" << exponent << " "<< d*pow(2,exponent)<< std::endl;    
  
  return d;
  
}


inline double ll2d(unsigned long long x) {
    union { double f; unsigned long long i; } tmp;
    tmp.i=x;
    return tmp.f;
  }
  
  
  inline unsigned long long d2ll(double x) {
    union { double f; unsigned long long i; } tmp;
    tmp.f=x;
    return tmp.i;
  }
  
inline
double cephes_log(double x){  
  
    double input_x=x;

    double y, z;

    double px,qx;

    /* separate mantissa from exponent */
    /* Note, frexp is used so that denormal numbers
    * will be handled properly.
    */
    
    unsigned long long n = d2ll(x);

    unsigned long long le = ((n >> 52) & 0x7ffL);
    int e = le;
    double fe =(e-1023);
    n &=0xfffffffffffffLL;
    constexpr unsigned long long p05 = (1022LL<<52); // d2ll(0.5);
    n |= p05;
    x = ll2d(n);
    if( x > SQRTH ) fe+=1.;
    if( x < SQRTH )   x += x;
    x =   x - 1.0;
   

    /* logarithm using log(x) = z + z**3 P(z)/Q(z),
    * where z = 2(x-1)/x+1)
    */
    /* logarithm using log(1+x) = x - .5x**2 + x**3 P(x)/Q(x) */
 
   

    /* rational form */

    z = x*x;
     px =  1.01875663804580931796E-4;
    px *= x;    
    px += 4.97494994976747001425E-1;
    px *= x;    
    px += 4.70579119878881725854E0;
    px *= x; 
    px += 1.44989225341610930846E1;
    px *= x; 
    px += 1.79368678507819816313E1;
    px *= x;
    px += 7.70838733755885391666E0;
    //
    //for the final formula
    px *= x; 
    px *= z;


    qx = x;
    qx += 1.12873587189167450590E1;
    qx *=x;
    qx += 4.52279145837532221105E1;
    qx *=x;    
    qx += 8.29875266912776603211E1;
    qx *=x;    
    qx += 7.11544750618563894466E1;
    qx *=x;    
    qx += 2.31251620126765340583E1;

    
    y = px / qx ;
   
    y -= fe * 2.121944400546905827679e-4; 
    y -= 0.5 * z  ;
    
    z = x + y;
    z += fe * 0.693359375;

    if (input_x > 5e307)
      z = std::numeric_limits<double>::infinity();
    if (input_x < 5e-307)
      z =  - std::numeric_limits<double>::infinity();       


    return( z );  
  
  }


int main(int argc, char** argv){

    const unsigned long long p05 = d2ll(0.5);
    printf("%X\n",p05); 
    std::cout << p05 << std::endl;
    std::cout << (1022LL<<52) << std::endl;

    double x= atof(argv[1]);

    std::cout << std::log(x) << " " << cephes_log(x) << std::endl;
    printf("%a %a\n",std::log(x),cephes_log(x));
    int e;
    std::cout << x << " " <<  frexp(x,&e) << " " << e <<  std::endl;

    double rndm_arr[30]={1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10};
    double cache[30];
    for (int i=0;i<30;++i)
      cache[i] = cephes_log(rndm_arr[i]);
    for (int i=0;i<30;++i)
      if (cache[i]==0.66666)
        std::cout << "pippo";
    
    
    
}


