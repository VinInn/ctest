
   // degree =  3   => absolute accuracy is
template<> inline float approx_atan2s_P< 3 >(float x){
  auto z = x*x;
  return  x * ((-10142.439453125f) + z * 2002.0908203125f);
}
// degree =  5   => absolute accuracy is
template<> inline float approx_atan2s_P< 5 >(float x){
  auto z = x*x;
  return   x * ((-10381.9609375f) + z * ((3011.1513671875f) + z * (-827.538330078125f)));
}
// degree =  7   => absolute accuracy is
template<> inline float approx_atan2s_P< 7 >(float x){
  auto z = x*x;
  return  x * ((-10422.177734375f) + z * (3349.97412109375f + z * ((-1525.589599609375f) + z * 406.64190673828125f))) ;
}
// degree =  9   => absolute accuracy is
template<> inline float approx_atan2s_P< 9 >(float x){
  auto z = x*x;
  return x * ((-10428.984375f) + z * (3445.20654296875f + z * ((-1879.137939453125f) + z * (888.22314453125f + z * (-217.42669677734375f)))));
}


template<int DEGREE>
inline short unsafe_atan2s_impl(float y, float x) {


  constexpr int maxshort = (int)(std::numeric_limits<short>::max())+1;
  constexpr short pi4 =  short(maxshort/4);
  constexpr short pi34 = short(3*maxshort/4);

  auto r= (std::abs(x) - std::abs(y))/(std::abs(x) + std::abs(y));
  if (x<0) r = -r;

  auto angle = (x>=0) ? pi4 : pi34;
  angle += short(approx_atan2s_P<DEGREE>(r));


  return (y < 0) ? - angle : angle ;

}

template<int DEGREE>
inline short unsafe_atan2s(float y, float x) {
  return unsafe_atan2s_impl<DEGREE>(y,x);
}
