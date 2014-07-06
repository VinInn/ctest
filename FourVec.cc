// typedef __complex__ double Value;
// typedef double Value;

// typedef __complex__ float Value;
typedef float Value;

struct LorentzVector
{

  LorentzVector(Value x=0, Value  y=0, Value  z=0, Value  t=0) : theX(x),theY(y),theZ(z),theT(t){} 
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
}  __attribute__ ((aligned(16)));

inline
int operator==(LorentzVector const & a, LorentzVector const & b) {
  int ret[4];
  ret[0] = a.theX==b.theX;ret[1] = a.theY==b.theY;ret[2] = a.theZ==b.theZ;ret[3] = a.theT==b.theT;
  return ret[0]&ret[1]&ret[2]&ret[3];
  //  return a.theX==b.theX & a.theY==b.theY & a.theZ==b.theZ & a.theT==b.theT;
}

inline LorentzVector
operator+(LorentzVector const & a, LorentzVector const & b) {
  return LorentzVector(a.theX+b.theX,a.theY+b.theY,a.theZ+b.theZ,a.theT+b.theT);
}

inline LorentzVector
operator-(LorentzVector const & a, LorentzVector const & b) {
  return LorentzVector(a.theX-b.theX,a.theY-b.theY,a.theZ-b.theZ,a.theT-b.theT);
}

inline LorentzVector
operator*(LorentzVector const & a, Value s) {
    return LorentzVector(a.theX*s,a.theY*s,a.theZ*s,a.theT*s);
}

inline LorentzVector
operator*(Value s, LorentzVector const & a) {
  return a*s;
}


inline Value dot(LorentzVector const & a, LorentzVector const & b) {
  return a.theX*b.theX+a.theY*b.theY+a.theZ*b.theZ-a.theT*b.theT;
}


//-----

void sum1(LorentzVector & res, Value s, LorentzVector const & v1, LorentzVector const & v2) {
  res += s*(v1+v2);
}


inline
LorentzVector __attribute__((always_inline)) ksum(Value s, LorentzVector const & v1, LorentzVector const & v2) {
  return s*(v1+v2) + s*(v1-2.f*v2) ;
}

void msum(LorentzVector & res, Value s, LorentzVector const & v1, LorentzVector const & v2) {
  res = s*(v1+v2) + s*(v1-2.f*v2) ;
}

namespace aos {
  LorentzVector a[1024], b[1014], c[1024];
  Value s;
  Value m[1024];
  void lsum() {
    for (int i=0; i!=1024;++i)
      a[i] = ksum(s,b[i],c[i]);
    for (int i=0; i!=1024;++i)
      m[i] =dot(a[i],b[i]);
  }
}

struct SoA4 {
  Value * mem;
  int n;
  Value x(int i) const  { return mem[i];} 
  Value y(int i) const { return  mem[i+n];} 
  Value z(int i) const { return  mem[i+2*n];} 
  Value t(int i) const { return  mem[i+3*n];} 
  Value & x(int i) { return mem[i];} 
  Value & y(int i) { return  mem[i+n];} 
  Value & z(int i) { return  mem[i+2*n];} 
  Value & t(int i) { return  mem[i+3*n];} 
  LorentzVector operator[](int i) const {
    return LorentzVector(x(i),y(i),z(i),t(i));
  }
  void set(LorentzVector const & v, int i) {
    x(i)=v.theX;y(i)=v.theY;z(i)=v.theZ;t(i)=v.theT;
  }
};

struct SoA3 {
  Value * mem;
  int n;
  Value x(int i) const  { return mem[i];} 
  Value y(int i) const { return  mem[i+n];} 
  Value z(int i) const { return  mem[i+2*n];} 

  Value & x(int i) { return mem[i];} 
  Value & y(int i) { return  mem[i+n];} 
  Value & z(int i) { return  mem[i+2*n];} 

  LorentzVector operator[](int i) const {
    return LorentzVector(x(i),y(i),z(i));
  }
  void set(LorentzVector const & v, int i) {
    x(i)=v.theX;y(i)=v.theY;z(i)=v.theZ;
  }
};

namespace soa4 {

  int N=1024;
  Value arena[3*4*1024];
  Value m1[4*1024],m2[4*1024],m3[4*1024];
  SoA4 a,b,c; 
  Value s;
  Value m[1024];
  void soAsum() {
    // a.mem=arena; b.mem=arena+4*1024;c.mem=b.mem+4*1024;
    a.mem=m1; b.mem=m2;c.mem=m3;
    a.n=b.n=c.n=1024;
    for (int i=0; i!=1024;++i)
      a.set(ksum(s,b[i],c[i]),i);
    for (int i=0; i!=1024;++i)
      m[i] =dot(a[i],b[i]);
  }
}

namespace soa3 {

  int N=1024;
  Value arena[3*3*1024];
  Value m1[3*1024],m2[3*1024],m3[3*1024];
  SoA3 a,b,c; 
  Value s;
  Value m[1024];
  void soAsum() {
    // a.mem=arena; b.mem=arena+4*1024;c.mem=b.mem+4*1024;
    a.mem=m1; b.mem=m2;c.mem=m3;
    a.n=b.n=c.n=1024;
    for (int i=0; i!=1024;++i)
      a.set(ksum(s,b[i],c[i]),i);
    for (int i=0; i!=1024;++i)
      m[i] =dot(a[i],b[i]);
  }
}
