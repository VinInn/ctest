#include<string>
class ExprEval {
public:
  ExprEval(const char * iname, const char * iexpr);
  ~ExprEval();
  
  template< typename EXPR>
  EXPR * expr() const { 
    typedef EXPR * factoryP();
    return reinterpret_cast<factoryP*>(m_expr)();
  }

private:

  std::string m_name;
  void * m_expr;
};
