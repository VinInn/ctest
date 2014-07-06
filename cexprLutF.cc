// constexpr int lutF(int i) { return i>3 ? 0 : 4-i;}

constexpr int lbit(int i, int n) {  return ( 0 ==(i >> 1) ) ? n :  lbit(i>>1, n+1);}
constexpr int lutF(int i) { return lbit(i,0);} 

float table[10];

float foo() {

  float const x = table[lutF(1)];
  float const y = table[lutF(2)];

  return x+y;

}

inline float t(int i) {
  return table[lutF(i)];
}

float foo2() {

  float const x = t(1);
  float const y = t(2);

  return x+y;

}


float bar() {

  constexpr int ix=lutF(1);
  float const x = table[ix];
  constexpr int iy=lutF(2);
  float const y = table[iy];

  return x+y;

}
