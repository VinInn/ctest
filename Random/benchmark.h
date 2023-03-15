#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <type_traits>
#include <chrono>
#include <iostream>
#include <algorithm>
#include<cmath>

namespace benchmark {

#ifdef __x86_64__
#define MAX_GPR_SIZE 8

  template <typename T>
  __attribute__((always_inline)) inline std::enable_if_t<sizeof(T) <= MAX_GPR_SIZE> keep(T&& x) noexcept {
    asm volatile("" : : "g"(x) :);
  }

  template <typename T>
  __attribute__((always_inline)) inline std::enable_if_t<(sizeof(T) > MAX_GPR_SIZE)> keep(T&& x) noexcept {
    asm volatile("" : : "m"(x) :);
  }

  template <typename T>
  __attribute__((always_inline)) inline std::enable_if_t<sizeof(T) <= MAX_GPR_SIZE> touch(T& x) noexcept {
    static_assert(!std::is_const<T>(), "touch argument is writeable");
    asm volatile("" : "+r"(x) : :);
  }

  template <typename T>
  __attribute__((always_inline)) inline std::enable_if_t<(sizeof(T) > MAX_GPR_SIZE)> touch(T& x) noexcept {
    static_assert(!std::is_const<T>(), "touch argument is writeable");
    asm volatile("" : "+m"(x) : :);
  }

#else
  // for other architecture one has to find the proper asm...
  template <typename T>
  keep(T&& x) noexcept {}
  template <typename T>
  touch(T& x) noexcept {}
#endif

  class TimeIt {
  public:
    using Clock = std::chrono::high_resolution_clock;
    using Delta = Clock::duration;
    using Time = Clock::time_point;

    TimeIt() { reset(); }

    void reset() {
      start = std::chrono::high_resolution_clock::now();
      delta = start - start;
    }

    template <typename T, typename U, typename F>
    void operator()(F f, T const* __restrict__ x, U* __restrict__ y, int n) {
      delta -= (std::chrono::high_resolution_clock::now() - start);
      touch(x);
      f(x, y, n);
      keep(y);
      delta += (std::chrono::high_resolution_clock::now() - start);
    }

    auto lap() const { return std::chrono::duration_cast<std::chrono::milliseconds>(delta).count(); }

  private:
    Time start;
    Delta delta;
  };

  template<int N>
  class Histo {
  public:
    using Self = Histo<N>;
    Histo(float mn, float mx) : xmin(mn), xmax(mx), ibsize(N/(xmax-xmin)){
     for ( auto & d : data) d=0;
    }

   void operator()(float x) {
     ++size;
     save+=x; svar+=x*x;
     int bin = std::clamp(int(ibsize*(x-xmin)),0,N-1);
     ++data[bin];
   }

   auto ave() const { return save/size;}
   auto var() const { return (svar-save*save)/(size-1.); }

   void printData(std::ostream & co) const {
     for (auto d : data) co << d <<',';
     co << std::endl;
   }

   template<typename F>
   void eval(F const & f, float * vf) const {
      float bsize =  (xmax-xmin)/N;
      float step = .05f*bsize;
      float x = xmin;
      constexpr float den = 1./19.;
      for (int i=0; i<N; ++i) {
        x+=step;
        float sum=0;
        for (int k=1;k<20;++k) { 
          sum += f(x); 
          x+=step;
        };
        vf[i] = bsize*size*sum*den; 
      };
   }

   template<typename F>
   void printAll(F const & f, std::ostream & co) const {
     float bsize =  (xmax-xmin)/N;
     float x = xmin+0.5f*bsize;
     float vf[N];
     eval(f,vf);
     std::cout << 'x' << std::endl;
     for (int i=0; i<N; ++i) {
        co << x <<',';
        x+=bsize;
     } 
     co << std::endl;
     std::cout << "data" << std::endl;
     for (auto d : data) co << d <<',';
     co << std::endl;
     std::cout << "expected" << std::endl;
     for (auto v : vf) co << v <<',';
     co << std::endl;
     std::cout << "expected error" << std::endl;
     for (auto v : vf) co << std::sqrt(v) <<',';
     co << std::endl;
   }

  template<typename F>
  double chi2(F const & f) const {
    double sum=0.;
    float vf[N];
    eval(f,vf);
    for (int i=0; i<N; ++i) {
      auto v = vf[i];;
      auto d = (v-data[i]);
      sum += d*d/v;  // error from prediction
    }
    return sum/(N-1);
  }


  void add(Self const & h) {
    for (int i=0; i<N; ++i) data[i]+=h.data[i];
    size+=h.size;
    save+=h.save;
    svar+=h.svar; 
  }

  private:
    uint64_t data[N];
    double size=0;
    double save=0;
    double svar=0;
    float xmin;
    float xmax;
    float ibsize;

  };

}  // namespace benchmark

#endif
