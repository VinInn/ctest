// SIMD dot product timings.
//
// Author: Emil Mikulic <emikulic@gmail.com>
// http://unix4lyfe.org/
//
#include <immintrin.h>
#include <cinttypes>
#include <cassert>
#include <cstdio>
#include <ctime>

static timespec mono_time() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t;
}

static timespec operator-(const timespec& a, const timespec& b) {
  timespec out;
  out.tv_sec = a.tv_sec - b.tv_sec;
  out.tv_nsec = a.tv_nsec - b.tv_nsec;
  if (out.tv_nsec < 0) {
    out.tv_sec -= 1;
    out.tv_nsec += 1000000000;
  }
  return out;
}

static void print(const timespec& t) {
  printf("%" PRId64 ".%09d", uint64_t(t.tv_sec), int(t.tv_nsec));
}

static_assert(sizeof(float) == 4, "");
static_assert(sizeof(__v4sf) == 16, "");

struct vec3f {
  float x;
  float y;
  float z;
};

float dot3_v1(const vec3f& a, const vec3f& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct vec4f {
  float x;
  float y;
  float z;
  float w;
};

// Generates identical code to dot3_v1:
// float dot3_v1_4(const vec4f& a, const vec4f& b) {
//   return a.x * b.x + a.y * b.y + a.z * b.z;
// }

float dot3_v2(const vec4f& a, const vec4f& b) {
  // This is more compact but also 128% slower than v1.
  __v4sf u = _mm_load_ps(&(a.x));
  __v4sf v = _mm_load_ps(&(b.x));
  // Mask:
  // 4 high bits: which elements should be summed. (w,z,y,x)
  // 4 low bits: which output slots should contain the result. (3,2,1,0)
  int mask = 0b01110001;
  return _mm_dp_ps(u, v, mask)[0];
}

float dot3_v3(const vec4f& a, const vec4f& b) {
  // This is 14% slower than v1.
  __v4sf u = _mm_load_ps(&(a.x));
  __v4sf v = _mm_load_ps(&(b.x));
  __v4sf s = _mm_mul_ps(u, v);
  return s[0] + s[1] + s[2];
}

// This function is here purely to break the optimizer by pretending to clobber
// the memory it's pointed at.  It compiles to a single "ret" instruction, and
// when inlined optimizes away into nothing.
void invalidate(void* ptr) {
  asm (
      "" /* no instructions */
      : /* no inputs */
      : /* output */ "rax"(ptr)
      : /* pretend to clobber */ "memory"
      );
}

int main() {
  const int times = 10;
  const int iters = 10000000;

  vec3f a3 = { 2, 3, 4 };
  vec3f b3 = { 6, 7, 8 };

  vec4f a4 = { 2, 3, 4, 5 };
  vec4f b4 = { 6, 7, 8, 9 };

#define TIME(fn, a, b) \
  for (int j = 0; j < times; ++j) {   \
    float f;                          \
    timespec t0 = mono_time(); \
    for (int i = 0; i < iters; ++i) { \
      invalidate(&a); \
      invalidate(&b); \
      f = fn(a, b); \
      invalidate(&f); \
    } \
    timespec t1 = mono_time(); \
    print(t1 - t0); \
    printf(" " #fn "\n"); \
    assert(f == 2*6 + 3*7 + 4*8); \
  }

  TIME(dot3_v1, a3, b3);
  TIME(dot3_v2, a4, b4);
  TIME(dot3_v3, a4, b4);

  return 0;
}

