#include "SharedQueue.h"
#include<cmath>
#include<random>
#include<memory>

struct Point {

  Point(double ix=0.,double iy=0.) : x(ix),y(iy){}
  double x;
  double y;

};

double distanceFast(Point const & rh, Point const & lh) {
  return std::max(std::abs(rh.x-lh.x),std::abs(rh.y-lh.y));
}

double distance2(Point const & rh, Point const & lh) {
  return std::pow((rh.x-lh.x),2) + std::pow((rh.y-lh.y),2);
}


struct Event {

  Event() {} 
  ~Event() {} 
  Event(Event && rh) : 
    hits(std::move(rh.hits)),
    seeds(std::move(rh.seeds)){}
  
  void swap(Event & rh) {
    std::swap(hits,rh.hits);
    std::swap(seeds,rh.seeds);
  }
  std::vector<Point> hits;
  std::vector<Point> seeds;
  std::vector<int> assoc;
  
};

std::vector<Point>::const_iterator closest(Point const & hit,  std::vector<Point>::const_iterator b, std::vector<Point>::const_iterator e) {
  if (b==e) return b;
  auto p = b;
  auto mind = distance2(hit,*b);
  while( (++b)!=e) {
    auto d = distance2(hit,*b);
    if (d<mind) {
      mind=d; p=b;
    }
  }
  return p;
}


void associate(Event & event) {
  event.assoc.resize(event.hits.size());
  for(int i=0;i!=event.hits.size(); ++i) 
    event.assoc[i] = closest(event.hits[i],event.seeds.begin(),event.seeds.end())-event.seeds.begin();
}

struct ClusterGenerator {

  explicit ClusterGenerator(float iclus, float ipoint) :
    rgen(-10.,10), clusGen(iclus), pointGen(ipoint), gauss(0.,1)
  {}
  
  std::unique_ptr<Event>  operator()() {
    std::unique_ptr<Event> event(new Event);

    int nclus = clusGen(reng);
    std::vector<Point> clus(nclus);
    std::vector<int> nhits(nclus);
    int tot=0;
    for (int i=0;i!=nclus; i++) 
      tot += nhits[i] = pointGen(reng);
    std::cout << nclus << " " << tot << std::endl;
    (*event).seeds.resize(nclus);
    (*event).hits.resize(tot);
    tot=-1; 
    for (int i=0;i!=nclus; ++i){
      clus[i].x = rgen(reng);
      clus[i].y = rgen(reng);
      for (int j=0; j!=nhits[i]; ++j) {
	++tot;
	(*event).hits[tot].x =  clus[i].x + gauss(reng);
	(*event).hits[tot].y =  clus[i].y + gauss(reng);
      }
      (*event).seeds[i].x = (*event).hits[tot].x;
      (*event).seeds[i].y = (*event).hits[tot].y;
    }
    return event;
  }
  
  std::ranlux_base_01 reng;
  std::uniform_real<float> rgen;
  std::poisson_distribution<float> clusGen;
  std::poisson_distribution<float> pointGen;
  std::normal_distribution<float> gauss;
  
  
};

void dump(Event const & event) {
  std::cout << "seeds #" << event.seeds.size() << std::endl;
  std::cout << "hits  #" << event.hits.size() << std::endl;
  for (int i=0;i!=event.seeds.size(); ++i)
    std::cout << '(' << event.seeds[i].x <<','
	      << event.seeds[i].y <<')' << ' ';
  std::cout << std::endl;

  for (int i=0;i!=event.hits.size(); ++i) {
    std::cout << '(' << event.hits[i].x <<','
	      << event.hits[i].y <<')' << ':'
	      << event.assoc[i] << ' ';
    if (i%10==9)   std::cout << std::endl;
  }
  std::cout << '\n' << std::endl;
}

int main(int argc, char * argv[]) {

  float nclus=10.;
  if (argc>1) nclus=::atof(argv[1]);
  float nhits=10.;
  if (argc>2) nhits=::atof(argv[2]);

  ClusterGenerator gen(nclus,nhits);
  std::unique_ptr<Event> event;
  event = gen();
  associate(*event);
  dump(*event);
  event = gen();
  associate(*event);
  dump(*event);
  // dump(*gen());

  return 0;

}
