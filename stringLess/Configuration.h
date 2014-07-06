#include "ConfigDefine.h"
#include "Config.h"



namespace {
  namespace c0N {

    static MyConfig<0> c;
    REGISTER(Producer0,c0_WithaLongTralingName_X,c)
    
    struct LocalWorkspace {
      LocalWorkspace() {
	c.whocares=3;
      }
    };
    
    static LocalWorkspace lw;
  }
}
