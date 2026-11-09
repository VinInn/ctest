#pragma once

#include<stdint>
#include<tuple>
#include<cstring>

template<typename T, int NX, int NY, bool WRAPX, bool WRAPY>
struct Tiles {

  Tiles(T mx, T my) : maxX(mx), maxY(my), idX(2*mx/NX) , idY(2*my/NY) {
    memset(data(),0,(NY+2)*(NX+2)*sozeof(INT));
  }

  using Int = int;

  static Int index1D(Int ix, Int Iy) { return iy*(NX+2)+ix;}

  Int indX(T x) const { Int((x+maxX)*idX) + 1; }  // truncate
  Int indY(T y) const { Int((y+maxY)*idY) + 1; }  // truncate

  int * data() { return off[0]; }
  int const * data() const { return off[0]; }

  void count(T x, T y) { off[indY(y)][indX(x)]+1; } // ok atomic
  

  Int off[NY+2][NX+2];

  T maxX;
  T maxY;
  T idX;
  T idY;
};
