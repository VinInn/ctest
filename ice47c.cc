typedef __complex__ double Value;
// typedef double Value;

// typedef __complex__ float Value;
// typedef float Value;

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
};  //  __attribute__ ((aligned(16)));

inline LorentzVector
operator+(LorentzVector a, const LorentzVector & b) {
  return a += b;
}

LorentzVector sum2(LorentzVector v1, LorentzVector v2) {
  return v1+v2;
}

void loop4(Value * __restrict__ x, Value * __restrict__ y, Value * __restrict__ z) {
  for(int i=0;i!=4;++i) 
    z[i] = x[i]+y[i];
} 

void loop(Value * __restrict__ x, Value * __restrict__ y, Value * __restrict__ z) {
  for(int i=0;i!=1024;++i) 
    z[i] = x[i]+y[i];
} 



Value ex, et;
LorentzVector sum() {
  LorentzVector v1; v1.theX =ex; v1.theY =ex+et; v1.theZ =ex-et;   v1.theT =et;
  return v1+v1;
}

