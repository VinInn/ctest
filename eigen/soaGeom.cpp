#include <cmath>
#include <cstdint>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <vector>
#include <memory>
#include <algorithm>

constexpr uint32_t stride() { return 5*1024;}
using V3 = Eigen::Vector3f;
using V15 = Eigen::Matrix<float,15,1>;
using MapV3 =  Eigen::Map<V3,0,  Eigen::InnerStride<stride()>>;
using MapV15 =  Eigen::Map<V15, 0, Eigen::InnerStride<stride()>>;


struct TSOS {
  V3 position;
  V3 momentum;
  V15 covariance;
};

struct TSOSsoa {
  explicit TSOSsoa(float * ip) :p(ip){}
  constexpr uint32_t posOff() {return 0;}
  constexpr uint32_t movOff() { return 3*stride();}
  constexpr uint32_t covOff() { return 6*stride();}

  auto position(uint32_t i) { return MapV3(p+posOff()+i);}
  auto momentum(uint32_t i) { return MapV3(p+movOff()+i);}
  auto covariance(uint32_t i)  { return MapV3(p+covOff()+i);}

  float * p;
};


struct Box {
  Eigen::AffineCompact3f transform;  
  V3 halfWidth;

  template<typename P3>
  inline bool inside(P3 const & v) const {
    return ((transform*v).array().abs() <  halfWidth.array()).all();
  }
};



constexpr uint32_t nTracks = 4096;
 
void doAOS(std::vector<TSOS> &trajs, Box const & b, std::vector<uint8_t> & res) {

  // std::vector<TSOS> trajs;
  // trajs.resize(nTracks);

  std::transform(trajs.begin(),trajs.end(),res.begin(),
		 [&](auto const& t){ return b.inside(t.position);});

}


void doSOA(TSOSsoa & trajSoa, Box const & b, uint8_t * res) {

  //  auto storage = std::make_unique<float[]>((3+3+15)*stride());
  // TSOSsoa trajSoa(storage.get());

  #pragma GCC ivdep
  for (auto i=0U; i<nTracks; ++i) {
    res[i] = b.inside(trajSoa.position(i));
  }
  
}


  
