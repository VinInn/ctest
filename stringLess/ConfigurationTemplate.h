#include "ConfigDefine.h"
#include "Config.h"

namespace {
  namespace c0N {

    static MyConfig<0> c;
    constexpr  Registerer::Key cName =  Registerer::hash("c0_WithaLongTralingName_X");
    constexpr  Registerer::Key cType =  Registerer::hash("Producer0");
    static Registerer  rc0(cType, cName, &c);
    
    struct LocalWorkspace {
      LocalWorkspace() {
	c.whocares=3;
      }
    };
    
    static LocalWorkspace lw;
  }
}
