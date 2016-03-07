
#include<cmath>

bool inside(float px, float py, float const * x, float const * y) {

   // bilinear interpolation of non-regular mesh: outside if does not map [0,1]

   auto C = (y[0] - py) * (x[3] - px) - (x[0] - px) * (y[3] - py);
   auto B = (y[0] - py) * (x[2] - px) + (y[1] - y[0]) * (x[3] - px) - 
            (x[0] - px) * (y[2] - y[3]) -(x[1] - x[0]) * (y[3] - py);
   auto A = (y[1] - y[0]) * (x[2] - x[3]) - (x[1] - x[0]) * (y[2] - y[3]);
   
   auto D = B * B - 4 * A * C;
   if (D<0) return false;

   // auto u = (-B - std::sqrt(D)) / (2.f * A);
   // if ( (u<0) | (u>1) ) return false;

  
  auto u = (-B - std::sqrt(D));
  auto d = 2.f*A;

  if (std::abs(u)>std::abs(d)) return false;
  u /=d;
  if (u<0) return false;
  

   auto p1x = x[0] + (x[1] - x[0]) * u;
   auto p2x = x[3] + (x[2] - x[3]) * u;
			
  //  auto v = (px - p1x) / (p2x - p1x);
   // return (v>=0) & (v<=1);

  return ( std::abs(px - p1x) <= std::abs(p2x - p1x) && (px - p1x)*(p2x - p1x) >=0 );
}




#include <iostream>


int main() {


  float x[4] = {0.5f, 1.0f,0.8f, 0.6};
  float y[4] = {0.5f, 0.6f,1.0f,0.8};

   auto go = [&](float xp, float yp,bool in){  
      auto is = inside(xp,yp,x,y);
      std::cout << (in ? "in " : "out " ) << xp <<',' << yp << " is " << (is ? "" :"not ") << "inside" << std::endl;
   };

   go(0,0,false); go(.55,.65,true), go(.51,.58,false);go(0.78,0.9,false);go(0.78,0.7,true);  
   go(1,1,false);

}
