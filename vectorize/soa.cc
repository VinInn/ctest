#include<cstdint>
struct Soa {

  uint32_t * mem;
  uint32_t ns;
  uint32_t cp;
  int const * __restrict__   i() const  __restrict__ { return (int const* __restrict__)(mem);}
  float const * __restrict__ f() const  __restrict__ { return (float const* __restrict__)(mem+cp);}
  float const * __restrict__ g() const  __restrict__ { return (float const* __restrict__)(mem+2*cp);}

};


void foo(Soa const &  __restrict__  soa, float * __restrict__ res) {
  for(std::size_t i=0; i!=soa.ns; ++i)
    res[i] = soa.f()[i]+soa.g()[i];
}

void bar(Soa const & __restrict__ soa, float * __restrict__ res) {
  float const * __restrict__ f = soa.f(); float const * __restrict__ g = soa.g();
  int n = soa.ns; for(int i=0; i!=n; ++i)
    res[i] = f[i]+g[i];
}


// inline
void add(float const * __restrict__ f, float const * __restrict__ g,float * __restrict__ res,int n) {
  for(int i=0; i!=n; ++i)
    res[i] = f[i]+g[i];
}

void add(Soa const & __restrict__ soa, float * __restrict__ res) {
   add(soa.f(),soa.g(),res,soa.ns);
}
