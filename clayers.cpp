#include<cstdint>
#include<array>

  template<class Function, std::size_t... Indices>
  constexpr auto make_array_helper(Function f, std::index_sequence<Indices...>) 
  -> std::array<typename std::result_of<Function(std::size_t)>::type, sizeof...(Indices)> 
  {
    return {{ f(Indices)... }};
  }

  template<int N, class Function>
  constexpr auto make_array(Function f)
  -> std::array<typename std::result_of<Function(std::size_t)>::type, N> 
  {
    return make_array_helper(f, std::make_index_sequence<N>{});    
  }


  constexpr uint32_t numberOfModules = 1856;

  constexpr uint32_t layerStart[11] = {0,96,320,672,1184,1296,1408,1520,1632,1744,1856};
  constexpr char const * layerName[10] = {"BL1","BL2","BL3","BL4",
                                          "E+1", "E+2", "E+3",
                                          "E-1", "E-2", "E-3"
                                          };

  constexpr uint32_t findMaxModuleStride() {
    bool go = true;
    int n=2;
    while (go) {
      for  (uint8_t i=1; i<11; ++i) {
        if (layerStart[i]%n !=0) {go=false; break;}
      }
      if(!go) break;
      n*=2;
    }
    return n/2;
  }

  constexpr uint32_t maxModuleStride = findMaxModuleStride();


  constexpr uint8_t findLayer(uint32_t detId) {
    for  (uint8_t i=0; i<11; ++i) if (detId<layerStart[i+1]) return i;
    return 11;
  }

  constexpr uint8_t findLayerFromCompact(uint32_t detId) {
    detId*=maxModuleStride;
    for  (uint8_t i=0; i<11; ++i) if (detId<layerStart[i+1]) return i;
    return 11;
  }


  /*
  constexpr std::array<uint8_t,numberOfModules> makeLayers() {
    std::array<uint8_t,numberOfModules>    res = {0};
    uint8_t l=0;
    for    (auto i=0U; i<numberOfModules; ++i) {
       if (layerStart[l]==i) ++l;
       res[i]=l-1;
    }
    return res;
  }
  */  

  constexpr uint32_t layerIndexSize = numberOfModules/maxModuleStride;
  constexpr std::array<uint8_t,layerIndexSize> layer = make_array<layerIndexSize>(findLayerFromCompact);

  constexpr bool validateLayerIndex() {
    bool res=true;
    for (auto i=0U; i<numberOfModules; ++i)  {
      auto j = i/maxModuleStride;
      res &=(layer[j]<10);
      res &=(i>=layerStart[layer[j]]);
      res &=(i<layerStart[layer[j]+1]);
    }
    return res;
  }

  static_assert(validateLayerIndex(),"Vincenzo's algo is buggy");

#include<iostream>
#include<cassert>
int main() {

  bool go = true;
  int n=2;
  while (go) { 
    for  (uint8_t i=1; i<11; ++i) {
      if (layerStart[i]%n !=0) {go=false; std::cout << int(i) << ' ' << layerStart[i] << ' ' << n << std::endl; break;}
    }
    if(!go) break;
    n*=2;
  }
  std::cout << "common factor "<< n/2 << std::endl;

  std::cout << "size of layer indeces " << layerIndexSize << std::endl;
  for (auto i=0U; i<numberOfModules; ++i)  {
    auto j = i/maxModuleStride;
    assert(layer[j]<10);
    assert(i>=layerStart[layer[j]]);
    assert(i<layerStart[layer[j]+1]);
  }
  return 0;

}
