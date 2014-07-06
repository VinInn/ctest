float bar(float const * x, float const * y,  int N) {

  constexpr float fact = .1e-6;
  
  auto f = [=](auto  a, auto b) {
    return a+ fact*b;
  };
  
  float r=0;
  for (int i=0; i<N; ++i) r+= f(x[i],y[i]);

  return r;
}
