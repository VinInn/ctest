#include <cmath> 
 
struct XYZ 
{ 
  float x; 
  float y; 
  float z; 
//  int w;
}; 
 
void 
norm (struct XYZ *in, struct XYZ *out, int size) 
{ 
#pragma GCC ivdep
  for (int i = 0; i < size; ++i) 
    { 
      float n = std::sqrt(in[i].x * in[i].x + in[i].y * in[i].y + in[i].z * in[i].z); 
      out[i].x = in[i].x / n;
      out[i].y = in[i].y / n;
      out[i].z = in[i].z / n;
    } 
} 
