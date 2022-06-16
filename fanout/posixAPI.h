#pragma once

#define __API__ posixN

#include "KernelFanout.h"


namespace posixN {
template<typename F, typename Tuple, std::size_t... Is>
void callIt(F f, launchParam const & p, const Tuple& t,std::index_sequence<Is...>){
  f(std::get<Is>(t)...);
}
}

#define __global__ inline
