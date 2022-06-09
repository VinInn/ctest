#pragma once

#include <hip/hip_runtime.h>

#define __API__ hipN

#include "KernelFanout.h"


namespace __API__ {
template<typename F, typename Tuple, std::size_t... Is>
void callIt(F f, launchParam const & p, const Tuple& t,std::index_sequence<Is...>){
  f<<<1,1,0,0>>>(std::get<Is>(t)...);
}
}

