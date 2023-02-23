#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <type_traits>
#include <chrono>

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

    template <typename T, typename F>
    void operator()(F f, T const* x, T* y, int n) {
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

}  // namespace benchmark

#endif
