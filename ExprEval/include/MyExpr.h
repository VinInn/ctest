#include <vector>
#include <algorithm>
#include <numeric>


#include "Cand.h"

struct MyExpr {
  using Coll = std::vector<Cand const*>; 
  using Res = std::vector<bool>;
  virtual void eval(Coll const &, Res &)=0;
  
};
