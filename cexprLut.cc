constexpr int lut[]={4,3,2,1};

float table[10];

float foo() {

  float const x = table[lut[1]];
  float const y = table[lut[2]];

  return x+y;

}

inline float t(int i) {
  return table[lut[i]];
}

float foo2() {

  float const x = t(1);
  float const y = t(2);

  return x+y;

}


float bar() {

  constexpr int ix=lut[1];
  float const x = table[ix];
  constexpr int iy=lut[2];
  float const y = table[iy];

  return x+y;

}
