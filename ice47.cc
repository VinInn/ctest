#include <complex>

template <typename Value> struct LorentzVector
{
template <typename ValueB>
LorentzVector<Value> & operator+=(const LorentzVector<ValueB> & a) {
theX += a.theX;
theY += a.theY;
theZ += a.theZ;
theT += a.theT;
return *this;
}

Value theX;
Value theY;
Value theZ;
Value theT;
};

template <typename ValueA, typename ValueB>
inline LorentzVector<ValueA>
operator+(LorentzVector<ValueA> a, const LorentzVector<ValueB> & b) {
return a += b;
}

LorentzVector<std::complex<double> > v1;
std::complex<double> evaluate() {
LorentzVector<std::complex<double> > veca = v1 + v1;
return veca.theT;
}

std::complex<double>  ex, et;
LorentzVector<std::complex<double> > sum() {
  LorentzVector<std::complex<double> > v1; v1.theX =ex; v1.theY =ex+et; v1.theZ =ex-et;   v1.theT =et;
  return v1+v1;
}
