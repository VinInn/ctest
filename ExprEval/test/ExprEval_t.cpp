#include "ExprEval.h"


int main() {

  std::string expr = "void myexpr::MyExpr::eval(Coll const &, Res &){}";

  ExprEval("MyExpr",expr.c_str());

  return 0;

}
