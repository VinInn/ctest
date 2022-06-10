#ifdef USE_CUDA
#include "cudaAPI.h"
#elif USE_HIP
#include "hipAPI.h"
#else
#include "posixAPI.h"
#endif
