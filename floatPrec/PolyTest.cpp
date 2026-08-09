// c++ -O3 benchCoth.cpp -I/data/innocent/benchmark/include /data/innocent/benchmark/build/src/libbenchmark.a -march=native -std=c++26 -pthread -lpfm
// ./a.out --benchmark_perf_counters=CYCLES,INSTRUCTIONS,RETIRED_FP_OPS_BY_TYPE:SCALAR_MAC,RETIRED_FP_OPS_BY_TYPE:SCALAR_ALL
// c++ -Ofast benchCoth.cpp -I/data/innocent/benchmark/include /data/innocent/benchmark/build/src/libbenchmark.a -march=native -std=c++26 -pthread -lpfm -funroll-loops -funroll-all-lo$
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

double ddd=0;
#include<iostream>
void end(benchmark::State& state) {
   using T = float;
   T sum=0;
    std::cout << ddd << std::endl;
   for (auto _ : state) {
     T x=T(-1);
     for (int i=0; i<4*1024*1024;i++) {
       sum+=x;
       x+=T(1.e-7);
     }
     benchmark::DoNotOptimize(sum);
   }
   add_IPC_counters(state);
   ddd+=sum;
   std::cout << ddd << std::endl;
}


// degree 6 polynomial (from "exp" expansion) 
template<bool ESTRIN>
inline float poly6(float y) {
constexpr float p[] = {float(0x2.p0),float(0x2.p0),float(0x1.p0),float(0x5.55523p-4),float(0x1.5554dcp-4),float(0x4.48f41p-8),float(0xb.6ad4p-12)};
if constexpr (!ESTRIN) {
  return  p[0] + 
        y * (p[1] + 
             y * (p[2] + 
                  y * (p[3] + 
                       y * (p[4] + 
                            y * (p[5] + 
                                 y * p[6]))))) ;
} else {// ESTRIN does seem to save a cycle or two
  float p56 = p[5] + y * p[6];
  float p34 = p[3] + y * p[4];
  float y2 = y*y;
  float p12 = p[1] + y; // By chance we save one operation here! Funny.
  float p36 = p34 + y2*p56;
  float p16 = p12 + y2*p36;
  float r =  p[0] + y*p16;
  return r;
}
}

template<bool ESTRIN>
void comp(float * b, float const * a, int N) {
  for (int i=0; i<N; ++i) {
    b[i] = poly6<ESTRIN>(a[i]);
  }
}

template<bool ESTRIN>
void loop(benchmark::State& state) {
   using T = float;
   T sum=0;
   for (auto _ : state) {
     benchmark::DoNotOptimize(sum);
//     for (T x=T(-1);  x<T(1); x+=T(1.e-7)) {
     T x=T(-1);
     for (int i=0; i<4*1024*1024;i++) {
//       benchmark::DoNotOptimize(x);
       // benchmark::DoNotOptimize(sum);
       sum += poly6<ESTRIN>(x);
       x+=T(1.e-7);
       // benchmark::DoNotOptimize(sum);
     }
     benchmark::DoNotOptimize(sum);
   }
   add_IPC_counters(state);
   ddd+=sum;
}


void horner(benchmark::State& state) { loop<false>(state);}
void estrin(benchmark::State& state) { loop<true>(state);}

BENCHMARK(horner);
BENCHMARK(estrin);


BENCHMARK(end);
BENCHMARK_MAIN();

