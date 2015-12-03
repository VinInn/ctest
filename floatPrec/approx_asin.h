#ifndef approx_asin_h
#define approx_asin_h

#include<cmath>
template<int DEGREE>
inline float approx_asin_P(float z);

   // degree =  3   => absolute accuracy is  8 bits
template<> inline float approx_asin_P< 3 >(float z){
 return  1.f + z * 0.2114248573780059814453125f;
}
   // degree =  5   => absolute accuracy is  12 bits
template<> inline float approx_asin_P< 5 >(float z){
 return  1.f + z * (0.1556626856327056884765625f + z * 0.1295671761035919189453125f);
}
   // degree =  7   => absolute accuracy is  15 bits
template<> inline float approx_asin_P< 7 >(float z){
 return  1.f + z * (0.1691854894161224365234375f + z * (5.1305986940860748291015625e-2f + z * 0.1058919131755828857421875f));
}
   // degree =  9   => absolute accuracy is  18 bits
template<> inline float approx_asin_P< 9 >(float z){
 return  1.f + z * (0.166119158267974853515625f + z * (8.322779834270477294921875e-2f + z * (5.28236292302608489990234375e-3f + z * 9.89462435245513916015625e-2f)));
}
   // degree =  11   => absolute accuracy is  21 bits
template<> inline float approx_asin_P< 11 >(float z){
 return  1.f + z * (0.1667812168598175048828125f + z * (7.249967753887176513671875e-2f + z * (6.321799755096435546875e-2f + z * ((-2.913488447666168212890625e-2f) + z * 9.9913299083709716796875e-2f))));
}


template<int DEGREE>
inline float unsafe_asin(float x) {
  auto z=x*x;
  return x*approx_asin_P<DEGREE>(z);
}

template<int DEGREE>
inline float unsafe_acos(float x) {
  constexpr float pihalf = M_PI/2; 
  return  pihalf - unsafe_asin<DEGREE>(x);
}

#endif
