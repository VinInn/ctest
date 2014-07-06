#include "vrandom.h"
#include "gaussian_ziggurat.h"
#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>
#include <array>
#include "vdtMath.h"

#include <iostream>

template <typename Iterator>
class Subiterator {
public:
  // Types
  
  typedef typename std::iterator_traits<Iterator>::iterator_category	iterator_category;
  typedef typename std::iterator_traits<Iterator>::value_type		value_type;
  typedef typename std::iterator_traits<Iterator>::difference_type	difference_type;
  typedef typename std::iterator_traits<Iterator>::pointer		pointer;
  typedef typename std::iterator_traits<Iterator>::reference		reference;
  
  // Constructors
  
  Subiterator () {}
  
  /** Subiterator p (pp, 3) provides an iterator which initially  has
   *  the same reference, but for which increments and offsets step by
   *  the amount stride rather than 1.  
   *  Thus p+k is equivalent to pp+(stride*k).
   * 
   *  Striding iterators are easily positioned beyond the bounds of the 
   *  underlying container.  It is up to the user to dereference the 
   *  iterator only when it has a valid reference.
   */
  Subiterator (const Iterator &iter, const difference_type& stride = 1)
    : _iter (iter), _stride (stride) {}
  
  template<class Iterator2>
  Subiterator (const Subiterator<Iterator2>& iter)
    : _iter (iter._iter), _stride (iter._stride) {}
  
  
  template<class Iterator2>
  Subiterator& operator = (const Subiterator<Iterator2>& sub)
  {
    _iter=sub._iter;
    _stride=sub._stride;
    return *this;
  }
  
  // Access operations
  reference operator * () const 
  { return *_iter; }
  
  Iterator operator -> () const 
  { return _iter; }
  
  reference operator [] (difference_type n) const 
  { return _iter[n * _stride]; }
  
  // Iteration operations
  
  Subiterator& operator ++ () 
  { _iter += _stride; return *this; }
  
  Subiterator operator ++ (int) 
  { Subiterator tmp = *this; _iter += _stride; return tmp; }
  
  Subiterator& operator -- () 
  { _iter -= _stride; return *this; }
  
  Subiterator operator -- (int) 
  { Subiterator tmp = *this; _iter -= _stride; return tmp; }
  
  Subiterator operator + (difference_type n) const 
  { return Subiterator (_iter + (n * _stride), _stride); }
  
  Subiterator& operator += (difference_type n) 
  { _iter += (n * _stride); return *this; }
  
  Subiterator operator - (difference_type n) const 
  { return Subiterator (_iter - (n * _stride), _stride); }
  
  difference_type operator - (const Subiterator& x) const 
  { return (_iter - x._iter)/_stride; }
  
  Subiterator& operator -= (difference_type n) 
  { _iter -= (n * _stride); return *this; }
  
  // Comparison operations
  
  bool operator == (const Subiterator& i) const 
  { return ( (this->_stride == i._stride) && (this->_iter == i._iter) ); }
  
  bool operator != (const Subiterator& i) const 
  { return !(*this == i); }
  
  bool operator < (const Subiterator& i) const 
  { return (this->_iter < i._iter); }
  
  bool operator > (const Subiterator& i) const 
  { return (this->_iter > i._iter); }
  
  bool operator <= (const Subiterator& i) const 
  { return (this->_iter <= i._iter); }
  
  bool operator >= (const Subiterator& i) const 
  { return (this->_iter >= i._iter); }
  
  void swap (Subiterator& x)
  { std::swap (_iter, x._iter); std::swap (_stride, x._stride); }
  
protected:
  
  Iterator	_iter;		// wrapped iterator
  difference_type	_stride;	// length between iterations
  
}; // template <class Iterator> class Subiterator


using Eng = vdt::MersenneTwister;

#include<iostream>

inline 
int knuthPoi(float l, Eng & eng) {
  auto cut = vdt::fast_expf(-l);
  int ret = 0;
  float x = 1.f;
  do {
    x*= eng.onef0();
    ++ret;
  } while (x>cut);
  return ret-1;
}


using I8 = std::array<int,8>;
using F8 = std::array<float,8>;


class VPoi {
public:

  static constexpr int Size = vdt::MersenneTwister::size();
  static constexpr int Lenght = Size/8;

  static constexpr int size() {return vdt::MersenneTwister::size(); }
  using Eng = vdt::MersenneTwister;

  using I8 = std::array<int,8>;
  using F8 = std::array<float,8>;
  using State= std::array<float,Size+8>;
  using Iter = Subiterator<std::array<float,Size>::iterator>;

  Eng eng;
  I8 ind;
  bool dogen=true;
  int loop=0;
  State state;
  State aux;
  int ia=7;

  void fill(State & r) {
      eng.generateState();
      for (int i=0; i!=8; ++i) r[i]=1.f;
      for (int i=8; i!=Size; ++i) r[i] = eng.fget0(i-8);
      for (int i=16; i!=Size; ++i) r[i]*=r[i-8];
  }

  int useAux(float c, float last) {
    dogen=true;
    if ((++ia)==8) {
      ia=0;
      fill(aux);
    }
    Iter b(8+aux.begin()+ia,8);
    Iter e(aux.end()-7+ia,8);
    auto p = std::lower_bound(b,e, c/last,std::greater<float>());
    
    std::cout << "aux " << c/last << " " << p-b << std::endl;
    return p-b;
  }

  inline
  void operator()(int * __restrict__ ret, F8 const & l, Eng & eng) {
    F8 cut;
    for (int i=0; i!=8; ++i) cut[i] =  vdt::fast_expf(-l[i]);

    //verify 
    for (int i=0; i!=8; ++i)
      if( (ind[i]>=Lenght) || cut[i]*state[8*ind[i]+i] < float(1.e-30) ) dogen=true;
    
    if(dogen) {
      eng.generateState();
      for (int i=0; i!=8; ++i) ind[i]=0;
      fill(state);
      dogen=false;
      loop=0;
    }
    
    
    
    for (int i=0; i!=8; ++i) {
      Iter b0(8+state.begin()+i,8);
      Iter b = b0+ind[i];
      Iter e(state.end()-7+i,8);
      Iter p=b;
      float c = (*(b-1))*cut[i];
      if (cut[i]>15.f) 
	p = std::lower_bound(b,e, c,std::greater<float>());
      else
	while (p!=e) { if (c> (*p)) break; ++p;}
      if (p!=e) {  ret[i] = p-b; ind[i]=p-b0; ind[i]++; }
      else ret[i] = p-b + useAux(c, *(e-1));
    } 

  }
    
};

unsigned int taux=0;
inline volatile unsigned long long rdtsc() {
 return __rdtscp(&taux);
}


inline double sum(int * v) {
  double sum=0;
  for (int i=0; i!=8*256; ++i) sum+=v[i];
  return sum;
}

inline double zero(int * v) {
  double sum=0;
  for (int i=0; i!=8*256; ++i) if (v[i]==0) sum++;
  return sum;
}


int main() {
  
  F8 k{0.5f,1.f,3.f,4.f,5.f,6.f,8.f,10.f};
  F8 l{2.5f,2.f,3.f,2.f,2.f,2.f,2.f,2.f};
  F8 h{10.5f,12.f,9.f,10.f,10.f,11.f,10.f,10.f};

  Eng eng;
 
 /*
  std::cout << eng.fmax() << " " << eng.norm() << std::endl;
  eng.generateState();
 
  constexpr int N = vdt::MersenneTwister::size();
  std::array<float,N> r;

  for (int i=0; i!=N; ++i) std::cout << eng.get(i)  << ",";
  std::cout << std::endl;

  for (int i=0; i!=N; ++i) r[i] = 0.5f*(eng.fget(i)+1.f);
  for (int i=0; i!=N; ++i) std::cout << r[i]  << ",";
  std::cout << std::endl;
  */


  std::cout << "size " << vdt::MersenneTwister::size() <<  std::endl;
  std::cout << "lenght " << vdt::MersenneTwister::size()/8 <<  std::endl;


  I8 res;
  for (int i=0; i!=8; ++i) 
    res[i] = knuthPoi(l[i],eng);

  for (auto r : res) std::cout << r << " ";
  std::cout << std::endl;

  VPoi vpoi;

  vpoi(res.begin(), l,eng);

  for (auto r : res) std::cout << r << " ";
  std::cout << std::endl;

 

  int arr[8*256];

  long long t1=0;
  float s1=0;
  float z1=0;
  long long t2=0;
  float s2=0;
  float z2=0;
  long long t3=0;
  float s3=0;
  float z3=0;
  long long t4=0;
  float s4=0;
  float z4=0;
  long long t5=0;
  float s5=0;
  float z5=0;
  long long t6=0;
  float s6=0;
  float z6=0;
  
  for (int i=0; i!=10000; ++i) {
    std::random_shuffle(k.begin(),k.end());
    std::random_shuffle(l.begin(),l.end());
    std::random_shuffle(h.begin(),h.end());


   t1 -= rdtsc();
    for (int j=0; j!=256*8; j+=8) 
      for (int n=0; n!=8; ++n) 
	arr[j+n] = knuthPoi(l[n],eng);
    t1 += rdtsc();
    s1+=sum(arr);
    z1+=zero(arr);
    
   t2 -= rdtsc();
   for (int j=0; j!=256*8; j+=8)
     vpoi(arr+j, l,eng);
   t2 += rdtsc();
   s2+=sum(arr);
   z2+=zero(arr);
 
   t3 -= rdtsc();
   for (int j=0; j!=256*8; j+=8) 
     for (int n=0; n!=8; ++n) 
       arr[j+n] = knuthPoi(h[n],eng);
   t3 += rdtsc();
   s3+=sum(arr);
   z3+=zero(arr);
   
   t4 -= rdtsc();
   for (int j=0; j!=256*8; j+=8)
     vpoi(arr+j, h,eng);
   t4 += rdtsc();
   s4+=sum(arr);
   z4+=zero(arr);
   
   t5 -= rdtsc();
   for (int j=0; j!=256*8; j+=8) 
     for (int n=0; n!=8; ++n) 
       arr[j+n] = knuthPoi(k[n],eng);
   t5 += rdtsc();
   s5+=sum(arr);
   z5+=zero(arr);
    
   t6 -= rdtsc();
   for (int j=0; j!=256*8; j+=8)
     vpoi(arr+j, k,eng);
   t6 += rdtsc();
   s6+=sum(arr);
   z6+=zero(arr);
   
   
   
  }
    
  std::cout << s1/(256*8)  << " " << z1/(256*8)  << " " << double(t1)/10000/(256*8) << std::endl;
  std::cout << s2/(256*8)  << " " << z2/(256*8)  << " " << double(t2)/10000/(256*8) << std::endl;
  std::cout << s3/(256*8)  << " " << z3/(256*8)  << " " << double(t3)/10000/(256*8) << std::endl;
  std::cout << s4/(256*8)  << " " << z4/(256*8)  << " " << double(t4)/10000/(256*8) << std::endl;
  std::cout << s5/(256*8)  << " " << z5/(256*8)  << " " << double(t5)/10000/(256*8) << std::endl;
  std::cout << s6/(256*8)  << " " << z6/(256*8)  << " " << double(t6)/10000/(256*8) << std::endl;

  return 0;

}

