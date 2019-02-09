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

template<int S>
using MapV3 =  Eigen::Map<V3,0,  Eigen::InnerStride<S>>;
template<int S>
using MapV15 =  Eigen::Map<V15, 0, Eigen::InnerStride<S>>;


template<int S>
struct BaseSOA {
  static constexpr uint32_t stride() { return S;}
  template<typename T>
  static constexpr uint32_t size() { return sizeof(T)*stride();}
  template<typename T>
  static constexpr uint32_t off(uint32_t preOff) { return preOff+size<T>();}
  
  //  char * p;
};


struct TSOS {
  V3 position;
  V3 momentum;
  V15 covariance;
  float charge;
};



template<int S>
struct TSOSsoa {
  static constexpr uint32_t stride() { return S;}
  template<typename T>
  static constexpr uint32_t size() { return sizeof(T)*stride();}

  static constexpr uint32_t posOff() {return 0;}
  static constexpr uint32_t movOff() { return size<V3>();}
  static constexpr uint32_t covOff() { return movOff()+size<V3>();}
  static constexpr uint32_t chargeOff() { return covOff()+size<V15>();}
  static constexpr uint32_t totSize() 
  { return chargeOff()+size<float>();}

  template<typename T>
  constexpr T * loc(uint32_t off, uint32_t i) {
    return ((T*)(data+off))+i;
  }
  
  auto position(uint32_t i) { return MapV3<S>(loc<float>(posOff(),i));}
  auto momentum(uint32_t i) { return MapV3<S>(loc<float>(movOff(),i));}
  auto covariance(uint32_t i)  { return MapV15<S>(loc<float>(covOff(),i));}
  auto charge(uint32_t i)  { return loc<float>(chargeOff(),i);}
  
  char data[totSize()];
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


void doSOA(TSOSsoa<maxN()> & trajSoa, Box const & b, uint32_t * res) {

  #pragma GCC ivdep
  for (auto i=0U; i<nTracks; ++i) {
    res[i] = b.inside(trajSoa.position(i));
  }
  
}
