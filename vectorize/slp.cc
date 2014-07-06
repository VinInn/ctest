typedef __complex__ float Value;
// typedef float Value;
// typedef double Value;

Value * __restrict__ x;
Value * __restrict__ y;
Value * __restrict__ z;

Value v1[4],v2[4],v3[4];

struct A {
  Value a[4];
}   __attribute__ ((aligned(16)));

A a1, a2, a3;

void foo ()
{

 Value * __restrict__ pin = &x[0];
 Value * __restrict__ pout =&y[0];

  *pout++ = *pin++;
  *pout++ = *pin++;
  *pout++ = *pin++;
  *pout++ = *pin++;
}

void voo() {

 Value * __restrict__ pin = &v1[0];
 Value * __restrict__ pout =&v2[0];

  *pout++ = *pin++;
  *pout++ = *pin++;
  *pout++ = *pin++;
  *pout++ = *pin++;


}

void aoo() {

 Value * __restrict__ pin = &a1.a[0];
 Value * __restrict__ pout =&a2.a[0];

  *pout++ = *pin++;
  *pout++ = *pin++;
  *pout++ = *pin++;
  *pout++ = *pin++;


}


void loop() {
  for (int i=0;i!=4;++i) x[i]=y[i]+z[i];
}

void voop() {
  for (int i=0;i!=4;++i) v1[i]=v2[i]+v3[i];
}

void aoop() {
  for (int i=0;i!=4;++i) a1.a[i]=a2.a[i]+a3.a[i];
}

void bar () {
  x[0]=y[0]+z[0];
  x[1]=y[1]+z[1];
  x[2]=y[2]+z[2];
  x[3]=y[3]+z[3];
}

void abar () {
  a1.a[0]=a2.a[0]+a3.a[0];
  a1.a[1]=a2.a[1]+a3.a[1];
  a1.a[2]=a2.a[2]+a3.a[2];
  a1.a[3]=a2.a[3]+a3.a[3];
}


A sum(A a, A b) {
  a.a[0]+=b.a[0];
  a.a[1]+=b.a[1];
  a.a[2]+=b.a[2];
  a.a[3]+=b.a[3];
  return a;
}


A suml(A a, A b) {
  for (int i=0;i!=4;++i) a.a[i]+=b.a[i];
  return a;
}


void dosum() {
  a1 = sum(a2,a3);
}

void dosuml() {
  a1 = suml(a2,a3);
}
