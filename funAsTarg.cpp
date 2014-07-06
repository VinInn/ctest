bool l(float a, float b) {
  return a<b;
}

bool l(double a, double b) {
  return a<b;
}

struct L{
  bool operator()(double a, double b) const {
    return a<b;
  }
};

template<typename T, typename F>
bool k(T a, T b, F const & f) {
  return f(a,b);
}

template<typename T, typename F>
bool j(T a, T b, F const & f) {
  return f(a,b);
}

template<typename T>
bool k(T a, T b, bool(*c)(T,T) ) {
  return c(a,b);
}


int main() {
  float a= 3.;
  double b = 3.;

  k(a,a,l);
  j(a,a,L());
  k(a,a,L());

  return 0;
}
