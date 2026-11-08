#pragma once

#include<stdint>
#include<tuple>

template<typename T, int NX, int NY, bool WRAPX, bool WRAPY>
struct Tiles {

  using Int = int;

  static Int index1D(Int ix, Int Iy) { return iy*(NX+2)+ix;}

  Int indX(T x) const { Int((x+maxX)*maxXinv) + 1; }  // truncate
  Int indY(T y) const { Int((y+maxY)*maxYinv) + 1; }  // truncate

  int * data() { return off[0]; }
  int const * data() const { return off[0]; }

  void count(T x, T y) { off[indY(y)][indX(x)]+1; } // ok atomic
  

  Int off[NY+2][NX+2];

  T maxX;
  T maxY;
  T maxXinv;
  T maxYinv;
};
