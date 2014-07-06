template<typename EXPR1, typename EXPR2>
struct op {
  op(EXPR1 const & il, EXPR2 const & ir) : l(il), r(ir){}
  EXPR1 const & l;
  EXPR2 const & r;
};

template<typename OP>
struct expr {
  expr(OP const & ie) : e(ie){}
private:
  expr(expr const &)=default;
  expr & operator=(expr const &)=default;
  expr(expr &&)=default;
  expr & operator=(expr &&)=default;

  template<typename E1, typename E2>
  friend expr<op<E1,E2>> bin(E1 const & e1, E2 const & e2);

  OP e;
};


template<typename E1, typename E2>
inline
expr<op<E1,E2>> bin(E1 const & e1, E2 const & e2) {
  return expr<op<E1,E2>>(op<E1,E2>(e1,e2));
} 


struct F {
template<typename EXPR>
    F(EXPR &&);
};

void bar() {
 using EI = expr<int>;
 int i=0;
 EI a(i),b(i),c(i);

 F f = bin(bin(a,b),c);

  // auto d = bin(bin(a,b),c);

  // auto e = d;    
}
