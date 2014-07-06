#ifndef X86TIME_H
#define X86TIME_H
    inline unsigned long long __attribute__((always_inline)) rdtsc(void)
    {
      unsigned hi, lo;
      __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
      return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
    }
#endif
