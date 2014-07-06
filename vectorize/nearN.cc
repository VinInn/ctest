#include<cmath>
#include<algorithm>

constexpr unsigned int NN=1024;
float eta[NN];
float phi[NN];
float distNN[NN];
int nn[NN];
int index[NN];

constexpr float pi=M_PI;
constexpr float twopi=2*pi;

inline float dist(int i, int j) {
  auto dphi = std::abs(phi[i] - phi[j]);
  auto deta = (eta[i] - eta[j]);
  dphi =  (dphi > pi) ? twopi - dphi : dphi;
  return (i==j) ? 100000.f : dphi*dphi + deta*deta;
}


inline float sdist(int i, int j) {
  auto dphi = (phi[i] - phi[j]);
  auto deta = (eta[i] - eta[j]);
  return dphi*dphi + deta*deta;
}

void nearNS(int j) {
  auto dold = distNN[j];
  auto ind = nn[j];
  for (int i=0; i!=NN; ++i) {
    auto d = sdist(i,j);
    ind = (d<dold)  ? index[i] : ind;
    dold = (d<dold)  ? d : dold;
    // dold = std::min(dold,d);
  }
  distNN[j] = dold;
  nn[j]=ind;
}


void nearN(int j) {
  for (int i=0; i!=NN; ++i) {
    auto dold = distNN[i];
    auto d = dist(i,j);
    nn[i]= (d<dold)  ? j : nn[i];
    distNN[i] = (d<dold)  ? d : dold;
  }

}


void nearNI(int j) {
  auto dold = distNN[j];
  auto ind = nn[j];
  for (int i=0; i!=NN; ++i) {
    auto d = dist(i,j);
    ind = (d<dold)  ? i : ind;
    // dold = (d<dold)  ? d : dold;
    dold = std::min(dold,d);
  }
  distNN[j] = dold;
  nn[j]=ind;
}


void nearNO(int j) {
  auto dold = distNN[j];
  auto ind = nn[j];
  for (int i=0; i!=NN; ++i) {
    auto d = dist(i,j);
    if (d<dold) {
      ind = i;
      dold =  d;
    } 
  }
  distNN[j] = dold;
  nn[j]=ind;
}




inline float dist2(int i, int j) {
  auto dphi = std::abs(phi[i] - phi[j]);
  auto deta = (eta[i] - eta[j]);
  dphi =  (dphi > pi) ? dphi = twopi - dphi : dphi;
  return dphi*dphi + deta*deta;
}

void nearN2(int j) {
  for (int i=0; i!=NN; ++i) {
    auto dold = distNN[i];
    auto d = (i==j) ? 100000.f : dist(i,j);
    nn[i]= (d<dold)  ? j : nn[i];
    distNN[i] = (d<dold)  ? d : dold;
  }

}
