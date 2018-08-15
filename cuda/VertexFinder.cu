#include<random>
#include<vector>
#include<cstdint>


struct Event {
  std::vector<float> zvert;
  std::vector<uint16_t>  itrack;
  std::vector<float> ztrack;
  std::vector<uint16_t> ivert;
};


struct ClusterGenerator {

  explicit ClusterGenerator(float nvert, float ntrack) :
    rgen(-13.,13), clusGen(nvert), trackGen(ntrack), gauss(0.,1.)
  {}

  void operator()(Event & ev) {

    int nclus = clusGen(reng);
    ev.zvert.resize(nclus);
    ev.itrack.resize(nclus);
    for (auto & z : ev.zvert) { 
       z = 13.0f*gauss(reng);
    }

    ev.ztrack.clear(); 
    ev.ivert.clear();
    for (int iv=0; iv<nclus; ++iv) {
      auto nt = trackGen(reng);
      ev.itrack[nclus] = nt;
      for (int it=0; it<nt; ++it) {
       ev.ztrack.push_back(ev.zvert[iv]+0.02f*gauss(reng));  // reality is not gaussian....
       ev.ivert.push_back(iv);
      }
    }
    // add noise
    auto nt = 2*trackGen(reng);
    for (int it=0; it<nt; ++it) {
      ev.ztrack.push_back(rgen(reng));
      ev.ivert.push_back(9999);
    }

  }

  std::mt19937 reng;
  std::uniform_real_distribution<float> rgen;
  std::poisson_distribution<int> clusGen;
  std::poisson_distribution<int> trackGen;
  std::normal_distribution<float> gauss;


};


#include<iostream>

int main() {


  Event  ev;

  ClusterGenerator gen(50,10);

  gen(ev);
  
  std::cout << ev.zvert.size() << ' ' << ev.ztrack.size() << std::endl;

}
