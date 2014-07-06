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
void
f20 (void)
{
  int i;
  for (i = 0; i < N; i++)
    g[i] = fx[k[i]]+fy[k[i]]+fz[k[i]];
}

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
float3 position[1024]; 


// int * __restrict__ neighList;
int neighList[1024];
int maxNeighbors;

float r2inv[1024];

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
}

void bar2(float3 ipos, int dis, int j, int i) {
  float * jp = &(position[0].x);
  
  for (int k=0; k<dis; k++){
    // auto in = neighList[k];
    auto in = neighList[j*dis + k + maxNeighbors * i];
    float delx = ipos.x -jp[3*in];
    float dely = ipos.y - jp[3*in+1];
    float delz = ipos.z - jp[3*in+2];
    r2inv[k] = delx*delx + dely*dely + delz*delz;
  }
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
}
