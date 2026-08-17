// c++ -O3 benchCoth.cpp -I/data/innocent/benchmark/include /data/innocent/benchmark/build/src/libbenchmark.a -march=native -std=c++26 -pthread -lpfm
// ./a.out --benchmark_perf_counters=CYCLES,INSTRUCTIONS,RETIRED_FP_OPS_BY_TYPE:SCALAR_MAC,RETIRED_FP_OPS_BY_TYPE:SCALAR_ALL
// c++ -Ofast benchCoth.cpp -I/data/innocent/benchmark/include /data/innocent/benchmark/build/src/libbenchmark.a -march=native -std=c++26 -pthread -lpfm -funroll-loops -funroll-all-loops
// ./a.out --benchmark_perf_counters=CYCLES,INSTRUCTIONS,RETIRED_FP_OPS_BY_TYPE:SCALAR_MAC,RETIRED_FP_OPS_BY_TYPE:SCALAR_ALL,RETIRED_FP_OPS_BY_TYPE:VECTOR_MAC

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

void ef(benchmark::State& state) { end<float>(state);}
void ed(benchmark::State& state) { end<double>(state);}


BENCHMARK(start);

BENCHMARK(nf);
BENCHMARK(nd);
BENCHMARK(sf);
BENCHMARK(sd);
BENCHMARK(pf);
BENCHMARK(pd);
BENCHMARK(af);
BENCHMARK(ad);
BENCHMARK(e16);

BENCHMARK(ef);
BENCHMARK(ed);


BENCHMARK_MAIN();

