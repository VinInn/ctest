#include "clockCuda.h"


struct Q {
   constexpr float operator()(float x) { return x+x*x;}
};

struct G {
  constexpr float operator()(int i) { return i*0.125e-4;}
};




int main() {

   doClock<G,Q,float>();

   return 0;
}
