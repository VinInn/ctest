enum class What {s,c};

template<What W>
struct DO {
static void doit(float k, float&o) {

  if constexpr (What::s==W)
    o = sinf(k);
  else
    o = cosf(k);

}
};


template <template<What> typename F> 
void do2(float k, float&o) {

  float a;
  F<What::s>::doit(k,o);
  F<What::c>::doit(k,a);
  o+=a;
}


void go(float k, float&o) {
 
   do2<DO>(k,o);

}

