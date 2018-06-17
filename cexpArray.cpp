#include <array>
#include <cmath>
constexpr 
std::array<float,18> cPhiGener() {
  std::array<float,18> cPhi{0}; 
  cPhi[0]=2.97025;
  for(unsigned i=1;i<=17;i++) cPhi[i]=cPhi[0]-2*i*M_PI/18;
  return cPhi;
}

constexpr 
std::array<float,18> cPhi = cPhiGener();


int main(int a, char **) {

  return a>2 ? cPhi[0] : cPhi[17];
  
}
