#ifndef ICSI_LOG_H
#define ICSI_LOG_H
#include<cmath>

/*
O. Vinyals, G. Friedland, and N. Mirghafori
ICSI Technical Report TR-07-002
http://www.icsi.berkeley.edu/cgi-bin/pubs/publication.pl?ID=002209
*/

/* Creates the ICSILog lookup table. Must be called
once before any call to icsi_log().
n is the number of bits to be taken from the mantissa
(0<=n<=23)
lookup_table is a pointer to a floating point array of 2^n positions.
*/
namespace icsi_details {

  template<int N>
  struct LookupTable {
    float * lookup_table;
    inline float operator[](int i) const { return lookup_table[i];}

    ~LookupTable() {delete []  lookup_table;}
    
    
    LookupTable() {
      fill_icsi_log_table();
    }
    
  void fill_icsi_log_table() {
    int n = N;
    union {
      float numlog;
      int x;
    } tmp;
    tmp.x = 0x3F800000; //set the exponent to 0 so numlog=1.0
    int incr = 1 << (23-n); //amount to increase the mantissa
    int p=std::pow(2.,n);
    lookup_table = new float[p];
    for(int i=0;i<p;++i)
      {
	lookup_table[i] = log2(tmp.numlog); //save the log value
	tmp.x += incr;
      }
    }
    
  };

}

/* Computes an approximation of log(val) quickly.
   val is a IEEE 754 float value, must be >0.
   lookup_table and n must be the same values as
   provided to fill_icsi_table.
   returns: log(val). No input checking performed.
*/
template<int N>
inline float icsi_log(float val,
		      icsi_details::LookupTable<N> const & lookup_table)
{
  const int n = N;
  union {
    float val;
    int x;
  } tmp;
  tmp.val=val;
  const int log_2 = ((tmp.x >> 23) & 255) - 127;//exponent
  tmp.x &= 0x7FFFFF; //mantissa
  tmp.x = tmp.x >> (23-n); //quantize mantissa
  val = lookup_table[tmp.x]; //lookup precomputed value
  return ((val + log_2)* 0.69314718f); //natural logarithm
}

namespace icsi_details {
  LookupTable<16> lookupTable16; 
  LookupTable<10> lookupTable10;
  LookupTable<5> lookupTable5; 
 }

float ln16(float x) {
  return icsi_log(x, icsi_details::lookupTable16);
}
float ln10(float x) {
  return icsi_log(x, icsi_details::lookupTable10);
}
float ln5(float x) {
  return icsi_log(x, icsi_details::lookupTable5);
}
 

#endif
