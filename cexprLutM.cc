struct P {
  int i;
  int v;
};
constexpr P lut[]={
  {1,4},
  {3,3},
  {2,0},
  {0,1},
  {-1,-1}
};

constexpr int lutF(int i,int j=0) {
  return lut[j].i < 0 ? lut[j].v : ( lut[j].i==i  ? lut[j].v : lutF(i,j+1));
}

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
