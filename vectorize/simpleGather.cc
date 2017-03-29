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

float3 f3[N];


void
aos (void)
{
  int i;
  for (i = 0; i < N; i++)
    g[i] = f3[k[i]].x+f3[k[i]].y+f3[k[i]].z;
}


// use gather
void
aos2 (void)
{
  float * ff = &(f3[0].x);
  int i;
  for (i = 0; i < N; i++)
    g[i] = ff[3*k[i]]+ff[3*k[i]+1]+ff[3*k[i]+2];
}


// use gather
void
soa (void)
{
  int i;
  for (i = 0; i < N; i++)
    g[i] = fx[k[i]]+fy[k[i]]+fz[k[i]];
}

