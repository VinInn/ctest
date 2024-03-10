#include<memory>
#include<tuple>
#include<cmath>

std::pair<double, double> rot(double phi, double x, double y) {
   double u = x*std::sin(phi) - y*std::cos(phi);
   double v = x*std::cos(phi) + y*std::sin(phi);

  return std::make_pair(u,v);
}




#include<cstdio>

int main(int n, char * arg[]) {

   double  fn = n;
   fn -= 1.;

   double phi = 1000*std::sqrt(2)+fn;
   double u, v;
   std::tie(u,v) = rot(phi,1.,1.);
  
  printf("%a %a %a\n",u,v,std::lgamma(n));

  return 0;

};
