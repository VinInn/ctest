// atom forces
struct float3 {
  float x;
  float y;
  float z;
};
 
#define N 4096
float fx[N], g[N];
float fy[N];
float fz[N]; 
int k[N];

float ff[3*N];
float3 f3[N];

float3 position[N]; 


int neighList[N];
int maxNeighbors;

float r2inv[N];
volatile float res;

void soaSeq(float3 ipos, int dis, int j, int i) {
  for (int k=0; k<dis; k++){
    auto in = k;
    float delx = ipos.x - fx[in];
    float dely = ipos.y - fy[in];
    float delz = ipos.y - fz[in];
    r2inv[k] = delx*delx + dely*dely + delz*delz;
 }
 res = r2inv[dis-1];
}


void seq(float3 ipos, int dis, int j, int i) {
  for (int k=0; k<dis; k++){
    auto in = k;
    float delx = ipos.x -position[in].x;
    float dely = ipos.y - position[in].y;
    float delz = ipos.z - position[in].z;
    r2inv[k] = delx*delx + dely*dely + delz*delz;
  }
  res = r2inv[dis-1];
}

// use gather
void soa(float3 ipos, int dis, int j, int i) {
  for (int k=0; k<dis; k++){
    auto in = neighList[k];
    // auto in = neighList[j*dis + k + maxNeighbors * i];
    float delx = ipos.x - fx[in];
    float dely = ipos.y - fy[in];
    float delz = ipos.y - fz[in];
    r2inv[k] = delx*delx + dely*dely + delz*delz;
 }
 res = r2inv[dis-1];
}

void aos(float3 ipos, int dis, int j, int i) {

  for (int k=0; k<dis; k++){
    auto in = neighList[k];
    // auto in = neighList[j*dis + k + maxNeighbors * i];
    auto jpos = position[ in ];
    float delx = ipos.x - jpos.x;
    float dely = ipos.y - jpos.y;
    float delz = ipos.y -jpos.z;
    r2inv[k] = delx*delx + dely*dely + delz*delz;
 }
 res = r2inv[dis-1];
}

// use gather instructions
void aos2(float3 ipos, int dis, int j, int i) {
  float * jp = &(position[0].x);
  
  for (int k=0; k<dis; k++){
    auto in = 3*neighList[k];
    // auto in = neighList[j*dis + k + maxNeighbors * i];
    float delx = ipos.x -jp[in];
    float dely = ipos.y - jp[in+1];
    float delz = ipos.z - jp[in+2];
    r2inv[k] = delx*delx + dely*dely + delz*delz;
  }
  res = r2inv[dis-1];
}

#include <iostream>
#include <chrono>
#include<algorithm>

void  time3(float3 const * ipos, int M, int dis, int j, int i) {

  // wakeup cpu
  for (int k=0; k<M; ++k) aos2(ipos[k], dis, j, i);


  auto start = std::chrono::high_resolution_clock::now();
  for (int k=0; k<M; ++k) soaSeq(ipos[k], dis, j, i);
  auto end = std::chrono::high_resolution_clock::now();
  auto deltas = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();

  start = std::chrono::high_resolution_clock::now();
  for (int k=0; k<M; ++k) seq(ipos[k], dis, j, i);
  end = std::chrono::high_resolution_clock::now();
  auto delta0 = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();

  start = std::chrono::high_resolution_clock::now();
  for (int k=0; k<M; ++k) soa(ipos[k], dis, j, i);
  end = std::chrono::high_resolution_clock::now();
  auto deltaa = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();
  

  start = std::chrono::high_resolution_clock::now();
  for (int k=0; k<M; ++k) aos(ipos[k], dis, j, i);
  end = std::chrono::high_resolution_clock::now();
  auto delta1 = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();

  start = std::chrono::high_resolution_clock::now();
  for (int k=0; k<M; ++k) aos2(ipos[k], dis, j, i);
  end = std::chrono::high_resolution_clock::now();
  auto delta2 = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();

  std::cout << "for dis " << dis << ' ' << deltas << ' ' << delta0 << ' ' << deltaa << ' ' << delta1 << ' ' << delta2 << std::endl;
 
}


int main() {
  int M = 100000;
  float3 ipos[M];

  std::cout << "tivially sequential" << std::endl;
  for (int i=0; i<N; ++i) 
     neighList[i] = i;
  for (int dis=2; dis<N; dis*=2) 
     time3(ipos,M,dis,0,0);

  std::cout << "small stride" << std::endl;
  for (int i=0; i<N; ++i)
     neighList[i] = (2*i)%N;
  for (int dis=2; dis<N; dis*=2)
     time3(ipos,M,dis,0,0);

  std::cout << "big stride" << std::endl;
  for (int i=0; i<N; ++i)
     neighList[i] = (32*i)%N;
  for (int dis=2; dis<N; dis*=2)
     time3(ipos,M,dis,0,0);


  std::cout << "random" << std::endl;
  std::random_shuffle(neighList,neighList+N);  
  for (int dis=2; dis<N; dis*=2)
     time3(ipos,M,dis,0,0);

  return 0;
}
