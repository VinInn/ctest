#include<cmath>
#include<algorithm>

constexpr unsigned int NN=1024;
float eta[NN];
float phi[NN];
float distNN[NN];
int nn[NN];
int index[NN];

inline float sdist(int i, int j) {
  auto dphi = (phi[i] - phi[j]);
  auto deta = (eta[i] - eta[j]);
  return dphi*dphi + deta*deta;
}


void selfLess(float & a, float b) {
  a = std::min(a,b);
}

#pragma omp declare reduction (nearN:float: selfLess(omp_out,omp_in))
#pragma omp declare reduction (nearN:int: omp_out = omp_in)

void nearNS(int j) {
  auto ne = distNN[j];
  auto ind = nn[j];
#pragma omp simd  reduction(min: ne) reduction(nearN: ind)
  for (int i=0; i<NN; ++i) {
    auto d = sdist(i,j);
    ind = (d<ne)  ? index[i] : ind;
    ne = (d<ne)  ? d : ne;
    // dold = std::min(dold,d);
  }
  distNN[j] = ne;
  nn[j]=ind;
}
