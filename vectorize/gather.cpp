// atom forces
struct float3 {
  float x;
  float y;
  float z;
};
 
#define N 1024
float fx[N], g[N];
float fy[N];
float fz[N]; 
int k[N];

float ff[3*N];
float3 f3[N];

// use gather
void
f20 (void)
{
  int i;
  for (i = 0; i < N; i++)
    g[i] = fx[k[i]]+fy[k[i]]+fz[k[i]];
}

// use gather
void
f21 (void)
{
  int i;
  for (i = 0; i < N; i++)
    g[i] = ff[3*k[i]]+ff[3*k[i]+1]+ff[3*k[i]+2];
}


void
f21b (void)
{
  int i;
  for (i = 0; i < N; i++) {
    auto j = ff+3*k[i];
    g[i] = j[0]+j[1]+j[2];
  }
}

void
f22 (void)
{
  int i;
  for (i = 0; i < N; i++)
    g[i] = f3[k[i]].x+f3[k[i]].y+f3[k[i]].z;
}


// position is an array of float3
// float3* __restrict__ position; 
float3 position[N]; 


// int * __restrict__ neighList;
int neighList[N];
int maxNeighbors;

float r2inv[N];
volatile float res;

void bar(float3 ipos, int dis, int j, int i) {

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
void bar2(float3 ipos, int dis, int j, int i) {
  float * jp = &(position[0].x);
  
  for (int k=0; k<dis; k++){
    auto in = neighList[k];
    // auto in = neighList[j*dis + k + maxNeighbors * i];
    float delx = ipos.x -jp[3*in];
    float dely = ipos.y - jp[3*in+1];
    float delz = ipos.z - jp[3*in+2];
    r2inv[k] = delx*delx + dely*dely + delz*delz;
  }
  res = r2inv[dis-1];
}


void foo(float3 ipos, int dis, int j, int i) {

  for (int k=0;k<dis;k++){ 
    auto in = neighList[k];
    // auto in = neighList[j*dis + k + maxNeighbors * i];

    auto jposx = position[ in ].x;
    auto jposy = position[ in ].y;
    auto jposz = position[ in ].z;
    
    
    float delx = ipos.x - jposx;
    float dely = ipos.y - jposy;
    float delz = ipos.z - jposz;
    r2inv[k] = delx*delx + dely*dely + delz*delz;  
 }
 res = r2inv[dis-1];
}


#include <iostream>
#include <chrono>
#include<algorithm>

void  time3(float3 const * ipos, int M, int dis, int j, int i) {

  // wakeup cpu
  for (int k=0; k<M; ++k) bar2(ipos[k], dis, j, i);
  
  auto start = std::chrono::high_resolution_clock::now();
  for (int k=0; k<M; ++k) bar(ipos[k], dis, j, i);
  auto end = std::chrono::high_resolution_clock::now();
  auto delta1 = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();

  start = std::chrono::high_resolution_clock::now();
  for (int k=0; k<M; ++k) bar2(ipos[k], dis, j, i);
  end = std::chrono::high_resolution_clock::now();
  auto delta2 = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();

  start = std::chrono::high_resolution_clock::now();
  for (int k=0; k<M; ++k) foo(ipos[k], dis, j, i);
  end = std::chrono::high_resolution_clock::now();
  auto delta3 = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();

  std::cout << "for dis " << dis << ' ' << delta1 << ' ' << delta2 << ' ' << delta3 << std::endl;

}


int main() {
  int M = 100000;
  float3 ipos[M];

  std::cout << "tivially sequential" << std::endl;
  for (int i=0; i<N; ++i) 
     neighList[i] = i;
  for (int dis=2; dis<N; dis*=2) 
     time3(ipos,M,dis,0,0);

  std::cout << "random" << std::endl;
  std::random_shuffle(neighList,neighList+N);  
  for (int dis=2; dis<N; dis*=2)
     time3(ipos,M,dis,0,0);

  return 0;
}
