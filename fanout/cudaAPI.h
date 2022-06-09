#pragma once

#include "cuda.h"

#define __API__ cudaN

#include "KernelFanout.h"


namespace __API__ {
template<typename F, typename Tuple, std::size_t... Is>
void callIt(F f, launchParam const & p, const Tuple& t,std::index_sequence<Is...>){
  f<<<1,1,0,0>>>(std::get<Is>(t)...);
}
}

