#include "ExprEval.h"

#include "MyExpr.h"


#include <iostream>

int main() {


  MyExpr::Coll c;
  for (int i=5; i<15; ++i) { c.emplace_back(new Cand(i,1,1)); }
  MyExpr::Res r;

  std::string expr = "void eval(Coll const & c, Res & r) override{ r.resize(c.size()); std::transform(c.begin(),c.end(),r.begin(), [](auto const & c){ return (*c).pt()>10;}); }";

  ExprEval parser("MyExpr",expr.c_str());

  auto func = parser.expr<MyExpr>();

  func->eval(c,r);

  std::cout << r.size()  << ' '  <<  std::count(r.begin(),r.end(),true) << std::endl;

  return 0;

}
