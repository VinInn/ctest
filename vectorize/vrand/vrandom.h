#ifndef VRANDOM_H
#define VRANDOM_H

#include <iosfwd>
#include <limits>
#include <type_traits>

#include <cstdint> // For uint_fast32_t, uint_fast64_t, uint_least32_t
#include<array>

namespace vdt {
/* Constant definitions! 
#define VEC_L	624				// Length of state-vector
#define SHF_C	397				// Shuffling constant
#define KNU_M	1812433253		// Knuth multiplier
#define ODG_C	2567483615		// Generator constant
#define TWC_1	2636928640		// Twisting constant #1
#define TWC_2	4022730752		// Twisting constant #2
*/
  namespace constantsMersenneTwister {
    //  Constant definitions! 
    constexpr unsigned int VEC_L = 624;			// Length of state-vector
    constexpr unsigned int SHF_C = 397;			// Shuffling constant
    constexpr unsigned int KNU_M = 1812433253;		// Knuth multiplier
    constexpr unsigned int ODG_C = 0x9908B0DFUL;		// Generator constant
    constexpr unsigned int TWC_1 = 0xffffffffUL;		// Twisting constant #1
    constexpr unsigned int TWC_2 = 0x9d2c5680UL;		// Twisting constant #2
    constexpr unsigned int TWC_3 = 0xefc60000UL;		// Twisting constant #3
    constexpr unsigned int default_seed = 5489u;

    inline unsigned int twiddle(unsigned int u, unsigned int v) {
      return (((u & 0x80000000UL) | (v & 0x7FFFFFFFUL)) >> 1)
      ^ ((v & 1UL) ? ODG_C : 0x0UL);
    };

    inline unsigned int twist(unsigned int y) {
      y ^= (y>>11) ;
      y ^= (y<< 7) & TWC_2;
      y ^= (y<<15) & TWC_3;
      y ^= (y>>18);
    
      return y;
    }

  }

  /* MersenneTwister class */
  class MersenneTwister{
  public:
    using index_type = unsigned int;
    using value_type = unsigned int;
    using signed_type = int;
    using float_type = float;
    using result_type = std::array<unsigned int,  constantsMersenneTwister::VEC_L>;
    using state_type = result_type;

  private:
    state_type state;
    unsigned int index;
    
    
  public:
    static inline constexpr index_type size() { return constantsMersenneTwister::VEC_L;}
    static inline constexpr value_type min()  { return  std::numeric_limits<value_type>::min();}
    static inline constexpr value_type max()  { return  std::numeric_limits<value_type>::max(); }
    static inline constexpr signed_type smax()  { return  std::numeric_limits<signed_type>::max(); }
    static inline constexpr float_type fmax()  { return  float(std::numeric_limits<signed_type>::max()); }
    static inline constexpr float_type norm()  { return  1.f/fmax(); }
    static inline constexpr float_type norm0()  { return  1.f/float(max()); }

    inline explicit MersenneTwister(unsigned int seed= constantsMersenneTwister::default_seed);

    inline void generateState();

    inline value_type get(index_type i) const { return constantsMersenneTwister::twist(state[i]); }

    // return rn [-1 1]
    inline float_type fget(index_type i) const { 
      union { value_type u; signed_type s; } w;
      w.u = get(i);
      return norm()*float(w.s);
    }

    // return rn [0 1]
    inline float_type fget0(index_type i) const { 
      return norm0()*float(get(i));
    }


    inline value_type one();

    inline float_type onef() { 
      union { value_type u; signed_type s; } w;
      w.u = one(); 
      return norm()*float(w.s);
    }

    inline float_type onef0() { 
      return norm0()*float(one());
    }


    inline result_type operator()();

    // calls f(x) N times where x is random in min and max above
    template< typename F>
    inline void loop(index_type N, F & fn) {
      using namespace constantsMersenneTwister;
      constexpr index_type state_size = VEC_L;

      /*
      if (index == VEC_L) generateState();
      index_type  lead = state_size - index;
      if (N<=lead) {
	for (index_type i=0; i!=N; ++i)  fn(twist(state[index+i]));
	index+=N;
	return;
      }
      
      for (index_type i=index; i!=state_size; ++i) fn(twist(state[i]));
      */
      index_type  lead = 0;

      N -= lead;
      index_type outLoop = N/state_size;
      for (index_type j=0; j!=outLoop; ++j)  {  
	generateState();
	for (index_type i=0; i!=state_size; ++i)  fn(twist(state[i]));
      }
      index_type trail = N - outLoop*state_size;
      if (trail) {
	generateState();
	for (index_type i=0; i!=trail; ++i)  fn(twist(state[i]));
	index = trail;
      }
    }
    
    inline void operator()(value_type __restrict__ * v, size_t vsz) {
      int k=0;
      auto fn =  [&v, &k](value_type r) { v[k++]= r;};
      loop(vsz,fn); 
    }

  };

  inline
  MersenneTwister::MersenneTwister(unsigned int seed) {
    using namespace constantsMersenneTwister;
    using tmp_type = unsigned int;
    tmp_type temporaryStateVector[VEC_L];
    index = 0;
    temporaryStateVector[0] = (tmp_type) seed;
    
    for ( index_type i=1; i<VEC_L; ++i)
      temporaryStateVector[i] =  ((KNU_M*( ( temporaryStateVector[i-1]^
					    (temporaryStateVector[i-1]>>30)
					    )
					   ) +i
				   )); // <<32)>>32;
    
    for (index_type i=0; i<VEC_L; i++)
      state[i] = (unsigned int) temporaryStateVector[i];

    index = VEC_L;
  }
  
  inline
  void MersenneTwister::generateState() {
    using namespace constantsMersenneTwister;


    constexpr index_type n = VEC_L;
    constexpr index_type m = SHF_C;
    
    for (index_type i = 0; i < (n - m); ++i)
      state[i] = state[i + m] ^ twiddle(state[i], state[i + 1]);
    
    for (index_type i = n - m; i < (n - 1); ++i)
      state[i] = state[i + m - n] ^ twiddle(state[i], state[i + 1]);
    
    state[n - 1] = state[m - 1] ^ twiddle(state[n - 1], state[0]);

    index = 0;
  }
  
  inline
  MersenneTwister::value_type
  MersenneTwister::one(){
    using namespace constantsMersenneTwister;
    if (index == VEC_L) generateState();
    
    return twist(state[index++]);
  }


  inline MersenneTwister::result_type 
  MersenneTwister::operator()() {
    using namespace constantsMersenneTwister;
    generateState();
    result_type r;
    for (index_type i = 0; i!=VEC_L; ++i)
      r[i]=twist(state[i]);
    return r;
  }
  
  
}  // vdt
#endif
