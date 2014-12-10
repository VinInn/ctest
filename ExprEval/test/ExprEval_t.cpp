#include "ExprEval.h"

#include "MyExpr.h"


#include <iostream>

int main() {


  MyExpr::Coll c;
  MyExpr::Res r;

  std::string expr = "void eval(Coll const & c, Res & r) override{ r.resize(10);}";

  ExprEval parser("MyExpr",expr.c_str());

  auto func = parser.expr<MyExpr>();

  func->eval(c,r);

  std::cout << r.size() << std::endl;

  return 0;

}
