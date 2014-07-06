#ifndef PerfTools_PentiumRealTime_H
#define PerfTools_PentiumRealTime_H

namespace perftools {

  inline unsigned long long int rdtsc() {
    unsigned long long int x;
    __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
    return x;
  }

}

#endif //  PerfTools_PentiumRealTime_H

