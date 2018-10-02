#ifndef memory_usage_h
#define memory_usage_h

#include <cstdint>

namespace memory_usage {
  bool     is_available();
  uint64_t allocated();
  uint64_t deallocated();
  uint64_t totlive();
};

#endif // memory_usage_h
