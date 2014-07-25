
#include "Base.h"
#include <cassert>
namespace {
   struct BAR { int i=0; int j=0;};

   thread_local BAR bar;

struct Me final : private Base {
  public:
   Me() { here(this);}

   int go(int i) const override { bar.i=i; return bar.j++; }

   int hi() const { return bar.i;}

};

  Me me;
}





