// typedef __complex__ float Value;
typedef __complex__ double Value;
// typedef float Value;
// typedef double Value;

struct A {
  Value a[4];
}   __attribute__ ((aligned(16)));

A a1, a2, a3;


A sum(A const & a, A const & b) {
  A ret;
  ret.a[0] = a.a[0]+b.a[0];
  ret.a[1] = a.a[1]+b.a[1];
  ret.a[2] = a.a[2]+b.a[2];
  ret.a[3] = a.a[3]+b.a[3];
  return ret;
}


A suml(A const & a, A const & b) {
  A ret;
  for (int i=0;i!=4;++i) ret.a[i]=a.a[i]+b.a[i];
  return ret;
}


void dosum() {
  a1 = sum(a2,a3);
}

void dosuml() {
  a1 = suml(a2,a3);
}


A sum2(A const & a1, A const & b) {
  A a = a1;
  a.a[0]+=b.a[0];
  a.a[1]+=b.a[1];
  a.a[2]+=b.a[2];
  a.a[3]+=b.a[3];
  return a;
}


A suml2(A a, A const & b) {
  for (int i=0;i!=4;++i) a.a[i]+=b.a[i];
  return a;
}
