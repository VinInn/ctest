#include "nativeVector.h"
#include "approx_vexp.h"
#include "approx_vlog.h"


int main() {


  using namespace nativeVector;

  std::cout << "native vector size is " << VSIZE << ' ' << sizeof(FVect)/sizeof(float) << std::endl;

  FVect z{0.f};
  
  std::cout << "zero " << z << std::endl; 


  FVect one = z+1.f;
  
  std::cout << "one " << one << std::endl; 

  IVect iz={0};

  IVect io = iz+1;

  std::cout << io <<  ' ' << iz << std::endl;
  std::cout << "-1?? " << ~iz << std::endl;


  std::cout << convert(io) << ' ' << convert(one) << std::endl;

  FVect x{-1.f,0.f,.34f,2.3};
  FVect a{1.f,123.f,31.f,1000.f};

  std::cout << x << " " << a << ' ' << (x>a) << std::endl; 
  if (testz(x>a)) std::cout << "all greater" << std::endl;
  
  std::cout << approx_expf<FVect,4,false>(x) << std::endl;
  std::cout << approx_logf<FVect,6>(a) << std::endl;
  std::cout << approx_logf<FVect,6>(approx_expf<FVect,4,false>(x)) << std::endl;
  std::cout << approx_expf<FVect,4,false>(approx_logf<FVect,6>(a)) << std::endl;

  float v[30]; for(int i=0;i<30;++i) v[i]=i+0.01f*i;
  IVect ind={0,3,4,2,15,6,6,3};
  FVect g = gather(v,ind);

  std::cout << "gather " << ind << ' ' << g << std::endl;
  
  return 0;
};
