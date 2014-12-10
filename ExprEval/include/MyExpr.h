#include <vector>
#include <algorithm>
#include <numeric>


#include "Cand.h"



namespace myexpr {


  struct MyExprBase {
    using Coll = std::vector<Cand const*>; 
    using Res = std::vector<bool>;
    virtual void eval(Coll const &, Res &)=0;

  };


  struct MyExpr final : public MyExprBase {
    void eval(Coll const &, Res &) override;
  };


}
