#include "ConfigDefine.h"
#include "Config.h"


template<int N>
struct Configurable {
  typedef MyConfig<N> Configuration;

  Configurable(void const * iparams) :
    params(*reinterpret_cast<Configuration const *>(iparmas)){}


  Configuration params;
};



struct Factory {



};
