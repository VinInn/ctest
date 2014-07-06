// #include<complex>


struct Complex {
  __complex__ double value;

  Complex() {
    __real__ value = 0;
    __imag__ value = 0;
  }
  Complex(Complex const & z)  {
    __real__ value = z.real();
    __imag__ value = z.imag();
  }
  Complex & operator=(Complex const & z) { 
    __real__ value = z.real();
    __imag__ value =z.imag();
    return *this;
  }
  Complex & operator+=(Complex const & z) {
    __real__ value += z.real();
    __imag__ value += z.imag();
    return *this;
  }
  
  double & real()  { return __real__ value;}
  double & imag()  { return __imag__ value;}

  double const & real() const { return __real__ value;}
  double const & imag() const { return __imag__ value;}

};

//typedef std::complex<double>  Value;
// typedef float Value;
// typedef double Value;
// typedef Complex Value;
typedef __complex__ double Value;
struct LorentzVector
{
  LorentzVector & operator+=(const LorentzVector & a) {
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

inline LorentzVector
operator+(LorentzVector a, const LorentzVector & b) {
  return a += b;
}

Value ex, et;
Value evaluate() {
  LorentzVector v1; v1.theX =ex; v1.theT =et;
  LorentzVector veca = v1 + v1;
  return veca.theT;
}

LorentzVector sum() {
  LorentzVector v1; v1.theX =ex; v1.theY =ex+et; v1.theZ =ex-et;   v1.theT =et;
  return v1+v1;
}

LorentzVector sum2(LorentzVector v1, LorentzVector v2) {
  return v1+v2;
}

LorentzVector operator*(LorentzVector v1, LorentzVector v2) {
  v1.theX*=v2.theX; v1.theY*=v2.theY; v1.theZ*=v2.theZ; v1.theT*=v2.theT;
  return v1;
}

Value dot(LorentzVector v1, LorentzVector v2) {
  return v1.theX*v2.theX+v1.theY*v2.theY+v1.theZ*v2.theZ+v1.theT*v2.theT;
}

Value dot2(LorentzVector v1, LorentzVector v2) {
  v1=v1*v2;
  return v1.theX+v1.theY+v1.theZ+v1.theT;
}
