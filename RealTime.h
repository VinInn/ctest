#ifndef PerfTools_RealTime_H
#define PerfTools_RealTime_H
//
//  defines "rdtsc"
//
#if defined(__i386__)

static __inline__ unsigned long long rdtsc(void)
{
  unsigned long long int x;
     __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
     return x;
}
#elif defined(__x86_64__)


static __inline__ unsigned long long rdtsc(void)
{
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

#elif defined(__powerpc__) || defined(__ppc__)


static __inline__ unsigned long long rdtsc(void)
{
  unsigned long long int result=0;
  unsigned long int upper, lower,tmp;
  __asm__ volatile(
                "0:                  \n"
                "\tmftbu   %0           \n"
                "\tmftb    %1           \n"
                "\tmftbu   %2           \n"
                "\tcmpw    %2,%0        \n"
                "\tbne     0b         \n"
                : "=r"(upper),"=r"(lower),"=r"(tmp)
                );
  result = upper;
  result = result<<32;
  result = result|lower;

  return(result);
}
#else
#error The file PerfTools/Measure/interface/RealTime.h needs to be set up for your CPU type.

#endif
/*
#if defined(__powerpc__) || defined(__ppc__) 
#include "PPCRealTime.h"
#elif defined(__i386__) || defined(__ia64) || defined(__ia64__) || defined(__x86_64__) || defined(__x86_64)
#include "PentiumRealTime.h"
#else
#error The file PerfTools/Measure/interface/RealTime.h needs to be set up for your CPU type.
#endif
*/

namespace perftools {

  typedef long long int TimeDiffType;
  typedef unsigned long long int TimeType;

  // High Precision real time in clock-units
  inline TimeType realTime() {
    return rdtsc();
  }

}


#endif //  PerfTools_RealTime_H
