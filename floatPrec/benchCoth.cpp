// c++ -O3 benchCoth.cpp -I/data/innocent/benchmark/include /data/innocent/benchmark/build/src/libbenchmark.a -march=native -std=c++26 -pthread -lpfm
// ./a.out --benchmark_perf_counters=CYCLES,INSTRUCTIONS,RETIRED_FP_OPS_BY_TYPE:SCALAR_MAC,RETIRED_FP_OPS_BY_TYPE:SCALAR_ALL
// c++ -Ofast benchCoth.cpp -I/data/innocent/benchmark/include /data/innocent/benchmark/build/src/libbenchmark.a -march=native -std=c++26 -pthread -lpfm -funroll-loops -funroll-all-loops
// ./a.out --benchmark_perf_counters=CYCLES,INSTRUCTIONS,RETIRED_FP_OPS_BY_TYPE:SCALAR_MAC,RETIRED_FP_OPS_BY_TYPE:SCALAR_ALL,RETIRED_FP_OPS_BY_TYPE:VECTOR_MAC
// ./a.out --benchmark_out=results.csv ; grep "cpu_time" results.csv | cut -f2 -d':' | tr '\n' ' '
#include <benchmark/benchmark.h>

void
add_flop_counters(benchmark::State &state, auto flop_per_iteration)
{
  state.counters["FLOP"] = {static_cast<double>(flop_per_iteration),
                            benchmark::Counter::kIsIterationInvariantRate};
  if (state.counters.contains("CYCLES"))
    state.counters["FLOP/cycle"] = {flop_per_iteration / state.counters["CYCLES"],
                                    benchmark::Counter::kIsIterationInvariant};
}

void
add_IPC_counters(benchmark::State &state)
{
  if (state.counters.contains("CYCLES") && state.counters.contains("INSTRUCTIONS"))
    state.counters["IPC"] = {state.counters["INSTRUCTIONS"] / state.counters["CYCLES"]};
}


#include<cmath>

template<typename T>
struct naive {
   using Float=T;
   T operator()(T x) { return std::sin(T(2)*std::atan(std::exp(x)));}
};

template<typename T>
struct secosh {
   using Float=T;
   T operator()(T x) { return T(1)/std::cosh(x);}
};


template<typename T>
struct poly {
   using Float=T;
   T operator()(T x) {
     x = std::abs(x)- T(0.6);
     return 
     T(0.999999986449) + x*(T(2.06961523705e-6) + x*( T(-0.500073072319) + x*( T(0.00106750423055) + x*( T(0.200707822535) + x*( T(0.0295750777896) + x*( T(-0.150307883454) + x*( T(0.0821370891714) + x*T(-0.0150543655971)
      )))))));
   }
};


template<typename T>
struct pade {
   using Float=T;
   T operator()(T x) {
     x = std::abs(x);
     T p = T(0.843551) + x*(T(-0.0488014) + x*(T(-0.0333239) + x*(T(0.00191492) + x*T(0.000527805) )));
     T q = T(1.0) + x*(T(0.479197) + x*(T(0.429426) + x*(T(0.0416364) + x*T(0.0185811)  )));
     return p/q;
   }
};

#include "Horner.h"

struct sech9 {
   using Float=float;
float operator()(float x) {
  constexpr float c[10] = {0.00484524607845, -0.0414213996033, 0.141985913707, -0.223653720345, 0.0820334481706, 0.178636110405, 0.00629169010252, -0.500693216667, 3.04210804006e-5, 0.999999773462};
  return horner<9>(x,c);
}
};

struct sech8 {
   using Float=float;
float operator()(float x) {
  constexpr float c[9] = {-0.00879617718167, 0.0505179227683, -0.0855020983306, -0.0394092119823, 0.24122061376, -0.0117719227185, -0.498093455092, -0.000112573426959, 1.00000105203};
  return horner<8>(x,c);
}
};

struct sech6 {
   using Float=float;
float operator() (float x) {
  constexpr float c[7] = {0.0325071921628, -0.183832500115, 0.340654638413, -0.0501527552854, -0.490379984351, -0.000761535402113, 1.00001123574};
  return horner<6>(x,c);
}
};

struct sech5 {
   using Float=float;
float operator() (float x) {
  constexpr float c[6] = {-0.0371366267403, 0.0924264377937, 0.143342769817, -0.558449420714, 0.00795787347873, 0.999832461682};
  return horner<5>(x,c);
}
};




float fin[1024];
double din[1024];
float fout[1024];
double dout[1024];

double dummy=0;

template<typename T>
struct data{
  static T *  in();
  static T * out();

};


template<>
struct data<float>{
  static float * in(){ return fin;}
  static float * out(){ return fout;}
};


template<>
struct data<double>{
  static double * in() { return din;}
  static double *  out() { return dout;}
};

#include<iostream>

void start(benchmark::State& state) {
   for (auto _ : state) {
     float x=-1.;
     for (int i=0; i<4*1024;i++) {
       for(int j=0; j<1024; ++j) { 
         fin[j]=x;
         din[j]=x; 
         x+=float(1.e-7);
       }
     }
     benchmark::DoNotOptimize(fin);
     benchmark::DoNotOptimize(din);
   }
   add_IPC_counters(state);
   std::cout << fin[100] << std::endl;
}
template<typename T>
void end(benchmark::State& state) {
   for (auto _ : state) {
     for (int i=0; i<4*1024;i++) {
       for(int j=0; j<1024; ++j) data<T>::out()[j]=data<T>::in()[j];
       benchmark::DoNotOptimize(data<T>::out());
     }
   }
   add_IPC_counters(state);
   std::cout << data<T>::out()[100] << std::endl;
}


template<typename F>
void loop(benchmark::State& state) {
   using T = F::Float;
//   using F = secosh<float>;
   F f;
   for (auto _ : state) {
     for (int i=0; i<4*1024;i++) {
       for(int j=0; j<1024; ++j) data<T>::out()[j]=f(data<T>::in()[j]);
       benchmark::DoNotOptimize(data<T>::out());
     }
   }
   add_IPC_counters(state);
}




#include "Exp16.h"
Exp16 exp16(5.);

void loop16(benchmark::State& state) {
   for (auto _ : state) {
     benchmark::DoNotOptimize(fout);
     uint16_t x = 0;
     for (int i=0; i<4*1024;i++) {
       for(int j=0; j<1024; ++j) {
         fout[j] =  2.f/(exp16.pexp(x)+exp16.nexp(x));
         x+=1;
       }
       benchmark::DoNotOptimize(fout);
       }
   }
   add_IPC_counters(state);
}


void e16(benchmark::State& state) { loop16(state);}


void nf(benchmark::State& state) { loop<naive<float>>(state);}
void nd(benchmark::State& state) { loop<naive<double>>(state);}

void sf(benchmark::State& state) { loop<secosh<float>>(state);}
void sd(benchmark::State& state) { loop<secosh<double>>(state);}

void pf(benchmark::State& state) { loop<poly<float>>(state);}
void pd(benchmark::State& state) { loop<poly<double>>(state);}

void af(benchmark::State& state) { loop<pade<float>>(state);}
void ad(benchmark::State& state) { loop<pade<double>>(state);}


void h5(benchmark::State& state) { loop<sech5>(state);}
void h6(benchmark::State& state) { loop<sech6>(state);}
void h8(benchmark::State& state) { loop<sech8>(state);}
void h9(benchmark::State& state) { loop<sech9>(state);}

void ef(benchmark::State& state) { end<float>(state);}
void ed(benchmark::State& state) { end<double>(state);}


BENCHMARK(start);

BENCHMARK(nd);
BENCHMARK(nf);
#ifdef ALL
BENCHMARK(sd);
#endif
BENCHMARK(sf);
#ifdef ALL
BENCHMARK(pd);
BENCHMARK(pf);
BENCHMARK(ad);
BENCHMARK(af);
#endif
BENCHMARK(h9);
#ifdef ALL
BENCHMARK(h8);
#endif
BENCHMARK(h6);
BENCHMARK(h5);
BENCHMARK(e16);

BENCHMARK(ef);
BENCHMARK(ed);


BENCHMARK_MAIN();

