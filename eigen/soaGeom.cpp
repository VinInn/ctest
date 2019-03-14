#include <cmath>
#include <cstdint>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <vector>
#include <memory>
#include <algorithm>

constexpr uint32_t maxN() { return 5*1024;}
using V3 = Eigen::Vector3f;
using V15 = Eigen::Matrix<float,15,1>;



struct TSOS {
  V3 position;
  V3 momentum;
  V15 covariance;
  float charge;
};


template<typename M, int S>
class alignas(128) MatrixSOA {
public:
  using Scalar = typename M::Scalar; 
  using Map = Eigen::Map<M, 0, Eigen::Stride<M::RowsAtCompileTime*S,S> >;
  using CMap = Eigen::Map<const M, 0, Eigen::Stride<M::RowsAtCompileTime*S,S> >;

  constexpr Map operator()(uint32_t i)  { return Map(data_+i);}
  constexpr CMap operator()(uint32_t i) const { return CMap(data_+i);}

private:
  Scalar data_[S*M::RowsAtCompileTime*M::ColsAtCompileTime];
  static_assert(sizeof(data_)%128==0);
};

template<typename M, int S>
class alignas(128) ScalarSOA {
  using Scalar = M; 

  constexpr Scalar & operator()(uint32_t i)  { return data_[i];}
  constexpr const Scalar operator()(uint32_t i) const { return data_[i];}

private:
  Scalar data_[S];
  static_assert(sizeof(data_)%128==0);
};


template<int S>
struct TSOSsoa {
  MatrixSOA<V3,S> position;
  MatrixSOA<V3,S> momentum;
  MatrixSOA<V15,S> covariance;
  ScalarSOA<float,S> charge;
};


struct Box {
  Eigen::AffineCompact3f transform; 
  V3 halfWidth;

  template<typename P3>
  constexpr uint32_t inside(P3 const & p) const {
    //auto r = (p).array().abs()-halfWidth.array();
    auto r = (transform*p).array().abs()-halfWidth.array();
    return (r(0)<0)+(r(1)<0)+(r(2)<0);
    // return ((transform*p).array().abs() <  halfWidth.array()).all();
  }
};



constexpr uint32_t nTracks = 4096;

void doAOS(std::vector<TSOS> &trajs, Box const & b, std::vector<uint8_t> & res) {

  std::transform(trajs.begin(),trajs.end(),res.begin(),
		 [&](auto const& t){ return b.inside(t.position);});

}


void doSOA(TSOSsoa<maxN()> const & trajSoa, Box const & b, uint32_t * res) {

  #pragma GCC ivdep
  for (auto i=0U; i<nTracks; ++i) {
    res[i] = b.inside(trajSoa.position(i));
  }
  
}
