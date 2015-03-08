#include<algorithm>
#include<iostream>
#include<tuple>


static constexpr unsigned int B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
// 01010101010101010101010101010101
// 00110011001100110011001100110011
// 00001111000011110000111100001111
// 00000000111111110000000011111111
static constexpr unsigned int S[] = {1, 2, 4, 8};



/* Interleave lower 16 bits of x and y, so the bits of x
 * are in the even positions and bits from y in the odd
 * z gets the resulting 32-bit Morton Number.
 * x and y must initially be less than 65536.
 * so better be unsigned short
 */
constexpr unsigned int zhash(unsigned int x, unsigned int y) {
  x = (x | (x << S[3])) & B[3];
  x = (x | (x << S[2])) & B[2];
  x = (x | (x << S[1])) & B[1];
  x = (x | (x << S[0])) & B[0];
    
  y = (y | (y << S[3])) & B[3];
  y = (y | (y << S[2])) & B[2];
  y = (y | (y << S[1])) & B[1];
  y = (y | (y << S[0])) & B[0];
    
  return y | (x << 1);

}


constexpr unsigned int  _000_ = 0;
constexpr unsigned int  _001_ = 1;
constexpr unsigned int  _010_ = 2;
constexpr unsigned int  _011_ = 3;
constexpr unsigned int  _100_ = 4;
constexpr unsigned int  _101_ = 5;

constexpr unsigned int  MASK = 0xaaaaaaaa; // hex(int('10'*10, 2))
constexpr unsigned int  XMASK = MASK;
constexpr unsigned int  YMASK = MASK>>1;


constexpr unsigned int  FULL = 0xffffffff;

constexpr unsigned int  BITMAX = 31;

constexpr unsigned int  setbits(unsigned int p, unsigned int v) {
  auto mask = (MASK >> (BITMAX-p)) & (~(FULL << p) & FULL);
  return  (v | mask) & ~(1 << p) & FULL;
}

constexpr unsigned int unsetbits(unsigned int p, unsigned int v) {
  auto mask = ~(MASK >> (BITMAX-p)) & FULL;
  return  (v & mask) | (1 << p);
}

// constexpr
inline
std::tuple<unsigned int,unsigned int> bigmin(unsigned int minz, unsigned int maxz, unsigned int zcode) {
  auto bigmin = maxz;
  auto litmax = minz;
  unsigned int bmx = __builtin_clz(((zcode^maxz)|(zcode^minz)));
  for (auto k=bmx; k<=BITMAX; k++) {
    auto p = BITMAX-k;
    auto mask = 1 << p;
    auto v = _000_;
    if (zcode & mask) v = _100_;
    if (minz & mask)  v |= _010_;
    if (maxz & mask)  v |= _001_;

    // std::cout << "cb " << p<< ' '<< v<< ' '<<minz<< ' '<<maxz<< ' ' << litmax << ' '<<bigmin << std::endl;
    
    if (v == _001_) {
      bigmin = unsetbits(p, minz);
      maxz = setbits(p, maxz);
    }  else if (v == _011_)
      return std::make_tuple(litmax,minz);
    else if (v == _100_)
      return std::make_tuple(maxz,bigmin);
    else if (v == _101_) {
      litmax = setbits(p, maxz);
      minz = unsetbits(p, minz);
    }
  }  
  return std::make_tuple(litmax,bigmin);
}


template<typename Iter>
Iter bisect(Iter a, Iter b, unsigned int zmin, unsigned int zmax) {
  return bisect(a,b,zmin,zmax,[](auto q) { return q;});
}
  
template<typename Iter, typename F>
Iter bisect(Iter a, Iter b, unsigned int zmin, unsigned int zmax, F f ) {
  auto e = b;
  while( (a<b) & ( f(*(b-1)) >= zmin )  ) {
    auto p = a+(b-a)/2;
    if ( f(*p)<zmin ) a=p;
    else if( f(*p)>zmax) b=p;
    else return p;
  }
  return e;
}


template<typename Iter, typename EX,  typename F, typename R>
void zsearch(Iter a, Iter b, unsigned int zmin, unsigned int zmax, EX ex, F f,
	     R range, unsigned int ozmin, unsigned int ozmax) {

#ifndef ZSEARCH_NO_LINEAR_OPT
  if (b-a < 64) {
    for (;a!=b;++a) if (range(ex(*a)))  f(*a); // report
    return;
  }
#endif
  
  auto p = bisect(a,b,zmin,zmax,ex);
  if (p==b) return;
  if (range(ex(*p))) {
    f(*p); // report
    zsearch(a,p,zmin,zmax,ex,f, range,ozmin,ozmax);
    zsearch(p+1,b,zmin,zmax,ex,f,range,ozmin,ozmax);
  } else {
    auto mxmn = bigmin(ozmin, ozmax, ex(*p));
    // std::cout << "cont " << ex(*p) << " " << zmin << ',' << std::get<0>(mxmn) << ' ' <<  std::get<1>(mxmn)  << ',' << zmax << std::endl;
    zsearch(a,p,zmin,std::get<0>(mxmn),ex,f,range,ozmin,ozmax);
    zsearch(p+1,b,std::get<1>(mxmn),zmax,ex,f,range,ozmin,ozmax);
  }

}


template<typename Iter, typename EX,  typename F>
void zsearch(Iter a, Iter b, unsigned int zmin, unsigned int zmax, EX ex, F f) {
  auto xmin = zmin&XMASK;
  auto ymin = zmin&YMASK;
  auto xmax = zmax&XMASK;
  auto ymax = zmax&YMASK;

  auto range = [=](auto z)->bool {
    auto x = z&XMASK;
    auto y = z&YMASK;
    return (x>=xmin)&(x<=xmax)&(y>=ymin)&(y<=ymax);
  };

  zsearch(a,b,zmin,zmax,ex,f, range, zmin, zmax);

}


#include<iostream>
#include<vector>
#include<cassert>
#include<random>
#include<functional>

void testBisect() {

  std::vector<unsigned int> v;

  auto p = bisect(v.begin(),v.end(),3,7);
  assert(p==v.end());

  v = {1};
  p = bisect(v.begin(),v.end(),3,7);
  assert(p==v.end());

  v = {1,2};
  p = bisect(v.begin(),v.end(),3,7);
  assert(p==v.end());


  v = {1,9};
  p = bisect(v.begin(),v.end(),3,7);
  assert(p==v.end());
  v = {1,2,9,12};
  p = bisect(v.begin(),v.end(),3,7);
  assert(p==v.end());

  
  v = {9};
  p = bisect(v.begin(),v.end(),3,7);
  assert(p==v.end());


  v = {9,11};
  p = bisect(v.begin(),v.end(),3,7);
  assert(p==v.end());

  
  v = {5};
  p = bisect(v.begin(),v.end(),3,7);
  assert(p==v.begin());


  v = {1,5};
  p = bisect(v.begin(),v.end(),3,7);
  assert(p==(v.begin()+1));

  v = {5,9};
  p = bisect(v.begin(),v.end(),3,7);
  assert(p==(v.begin()));


  v = {1,5,9};
  p = bisect(v.begin(),v.end(),3,7);
  assert(p==(v.begin()+1));

  v = {1,3,5,9};
  p = bisect(v.begin(),v.end(),3,7);
  assert( (p>v.begin()) & (p<v.end()));

  v = {4,5};
  p = bisect(v.begin(),v.end(),3,7);
  assert(p!=v.end());

  
}

#include <x86intrin.h>

unsigned int taux=0;
inline volatile unsigned long long rdtsc() {
 return __rdtscp(&taux);
}


void testZSearch(int N) {

  std::default_random_engine generator;
  std::uniform_int_distribution<unsigned int> distribution(0,0xffff);
  auto pos = std::bind ( distribution, generator );

  int k=0; unsigned int m=0;
  long long t1=0, t2=0, t3=0;


  for (int j=0; j<100; ++j) {
  struct X{  unsigned int x, y, z; };
  std::vector<X> w;

  for (int i=0; i<N; ++i) {
    auto x=pos(); auto y=pos();
    w.emplace_back(X{x,y,zhash(x,y)});
  }
  std::sort(begin(w),end(w),[](auto const & a, auto const & b) { return a.z<b.z;});
  // std::cout << w[0].z << ' ' << w.back().z << std::endl;
  auto w2 = w;
  std::sort(begin(w2),end(w2),[](auto const & a, auto const & b) { return a.x<b.x;});
  
  

  for (unsigned int i=10; i<5000; ++i) {
    auto x=pos(); auto y=pos(); auto mx = 0xffff -i;
    auto xmin = (x<i) ? 0 : x-i;
    auto ymin = (y<i) ? 0 : y-i;
    auto xmax = (x>mx) ?  0xffff : x+i;
    auto ymax = (y>mx) ?  0xffff : y+i;
    auto zmin = zhash(xmin, ymin);
    auto zmax = zhash(xmax, ymax);
    assert(xmin<0xffff);
    assert(ymin<0xffff);
    assert(xmax<=0xffff);
    assert(ymax<=0xffff);
    
    auto range = [=](auto x, auto y)->bool {
      return (x>=xmin)&(x<=xmax)&(y>=ymin)&(y<=ymax);
    };


    // brute force
    std::vector<unsigned int> s0; s0.reserve(N);
    t1 -= rdtsc();
    std::for_each(begin(w),end(w),[&](auto z) {
	if (range(z.x,z.y)) s0.push_back(z.z);}
      );
    t1 += rdtsc();

    // x sorting
    std::vector<unsigned int> s2; s2.reserve(N);
    t2 -= rdtsc();
    X wmin{xmin,ymin,zmin};
    X wmax{xmax,ymax,zmax};
    auto a = std::lower_bound(begin(w2),end(w2),wmin,[&](auto x1, auto x2) { return x1.x < x2.x;});
    auto b = std::upper_bound(a,end(w2),wmax,[&](auto x1, auto x2) { return x1.x < x2.x;});
    std::for_each(a,b,[&](auto z) {
	if (range(z.x,z.y)) s2.push_back(z.z);}
      );
   t2 += rdtsc();
   std::sort(begin(s2),end(s2));
   assert(s0==s2);
     

    // zseach
    std::vector<unsigned int> s1; s1.reserve(N);
    t3 -= rdtsc();
    zsearch(w.begin(),w.end(),zmin,zmax,[](auto q) { return q.z;},
	    [&](auto z) {s1.push_back(z.z);}
	    );
    t3 += rdtsc();
    std::sort(begin(s1),end(s1));

    assert(s0==s1);
    if (s0.empty()) k++;
    if (s0.size()>m) m = s0.size();
  }

  }
  std::cout << N << ' ' << k << ' ' << m <<std::endl;
  std::cout << double(t1)/1000000. << ' ' << double(t2)/1000000. << ' ' << double(t3)/1000000. << std::endl;
  
}


int main() {


  testBisect();


  for (int N=32; N<17000; N*=2)
    testZSearch(N);


  
  std::cout << zhash(1,1) << std::endl;
  std::cout << zhash(5,1) << std::endl;
  std::cout << zhash(1,5) << std::endl;

  std::cout << zhash(3,5) << std::endl;
  std::cout << zhash(0,5) << std::endl;
  std::cout << zhash(3,0) << std::endl;
  std::cout << (zhash(3,5)&XMASK) << std::endl;
  std::cout << (zhash(3,5)&YMASK) << std::endl;


  std::cout << zhash(5,5) << std::endl;
  std::cout << (zhash(5,5)&XMASK) << std::endl;
  std::cout << (zhash(5,5)&YMASK) << std::endl;


  auto xmin = 3U;
  auto ymin = 5U;
  auto xmax = 6U;
  auto ymax = 10U;
  auto zmin = zhash(xmin, ymin);
  auto zmax = zhash(xmax, ymax);

  auto range = [=](auto x, auto y)->bool {
    return (x>=xmin)&(x<=xmax)&(y>=ymin)&(y<=ymax);
  };


  unsigned int z = 58;


  
  auto mxmn = bigmin(zmin, zmax, z);
  std::cout << zmin << ' ' << zmax << ' ' << std::get<0>(mxmn) << ',' <<  std::get<1>(mxmn) << std::endl;

  
  struct X{  unsigned int x, y, z; };
  std::vector<X> w;
  auto k=0U;
  for  ( auto x=0U; x<=9; ++x)
    for ( auto y=0U; y<=17; ++y)
      if ( (++k)%3==0 ) w.emplace_back(X{x,y,zhash(x,y)});


  auto dotest = [&]() {
  
    std::sort(begin(w),end(w),[](auto const & a, auto const & b) { return a.z<b.z;});
    std::cout << w[0].z << ' ' << w.back().z << std::endl;
    
    int k1=0;
    std::for_each(begin(w),end(w),[&](auto z) {
	if (range(z.x,z.y)) std::cout << "found " <<k1++ << ' ' << z.z << ':' << z.x << ',' << z.y << std::endl;}
      );
    
    
    auto p = bisect(w.begin(),w.end(),zmin,zmax,[](auto q) { return q.z;});
    if (p!=w.end()) {
      auto z = *p;
      std::cout << w[0].z << ' ' << w.back().z << ' ' << z.z << ':' << z.x << ',' << z.y << std::endl;
      auto mxmn = bigmin(zmin, zmax, z.z);
      std::cout << zmin << ' ' << zmax << ' ' << std::get<0>(mxmn) << ',' <<  std::get<1>(mxmn) << std::endl;
    }  
    
    int k2=0;
    zsearch(w.begin(),w.end(),zmin,zmax,[](auto q) { return q.z;},
	    [&](auto z) { std::cout << "found " << k2++ << ' ' << z.z << ':' << z.x << ',' << z.y << std::endl;}
	    ); 

    assert(k1==k2);
  };  // end dotest

  
  dotest();
  w.emplace_back(X{3,3,zhash(3,3)});
  w.emplace_back(X{3,3,zhash(3,3)});
  w.emplace_back(X{3,3,zhash(3,3)});
  dotest();
  w.emplace_back(X{4,7,zhash(4,7)});
  w.emplace_back(X{4,7,zhash(4,7)});
  w.emplace_back(X{4,7,zhash(4,7)});
  dotest();
  
  

  
  return 0;

}
