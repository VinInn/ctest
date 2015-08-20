#include<iostream>
#include<iomanip>
#include<cmath>
#include<limits>
#include<cstdio>

template<typename T> 
void print(T x) {
 std::cout<< std::hexfloat << x <<' '<<  std::scientific << std::setprecision(8) << x << ' ' <<  std::defaultfloat << x << std::endl;
}

template<typename T>
void cprint(T x) {printf("%a %11.8e %f\n",x,x,x);}
  

void dprint(double x) {printf("%a %19.16e %f\n",x,x,x);}



int main() {
  
  dprint(M_PI);
  float pi = std::acos(-1);
  print(pi);
  cprint(pi);
  print(std::nextafter(pi,2.f));
  print(std::nextafter(pi,4.f));

  dprint(0x1.921fb6p+1);
  dprint(0x1.921fb5p+1);
  dprint(0x1.921fb7p+1);


  std::cout << std::endl;
  float tenth = 0.1;
  float one = 1.;
  print(tenth);
  cprint(tenth);
  auto next = std::nextafter(tenth,1.f);
  print(next);
  cprint(next);

  auto ulp = (next-tenth);
  print (ulp);
  auto prec = ulp/tenth;
  print (prec);

  long long large = 1000*1000*1000;
  std::cout << large << std::endl;
  float flarge = large;
  print(flarge);
  print(flarge+32.f-flarge);
  print(flarge+33.f-flarge);

  dprint(0x1.dcd650p+29);
  dprint(0x1.dcd64ep+29);
  dprint(0x1.dcd64fp+29);
  dprint(0x1.dcd651p+29);
  dprint(0x1.dcd652p+29);
  auto garge = large;
  while ( float(--garge)==flarge );
  float nlarge = garge;
  std::cout << garge << std::endl;
  print(nlarge);
  print((nlarge-flarge)/flarge);

  while ( float(++large)==flarge );
  nlarge = large;
  std::cout << large << std::endl;
  print(nlarge);
  print((nlarge-flarge)/flarge);


  while ( float(++large)==nlarge );
  nlarge = large;
  std::cout << large << std::endl;
  print(nlarge);
  print((nlarge-flarge)/flarge);

  std::cout << std::endl;   
  float x = 1000*1000*1000;
  std::cout <<  std::scientific << std::setprecision(8) << x << ' ' << x+32.f << ' ' << x+33.f << std::endl;

  auto nf = [](auto a, auto b) {
    long long n=0;
    while(a!=b) { a=std::nextafter(a,b); ++n;}
    std::cout << n << std::endl;
  };

  nf(0.1f,0.2f); 
  nf(1000.f,2000.f);


  std::cout << std::endl;
  float x1=0;
  for (float y=0;y<=1000000; ++y)x1+=y;
  print (x1);

  float result = 0.5*1000000 *(1000000+1);

  
  float x2=0;
  for (float y=1000000;y>0; --y)x2+=y;
  print (x2);
  print (x2-x1); print ((x2-x1)/x1);

  std::cout << std::scientific << (result-x1)/result << std::endl;
  std::cout << std::scientific << (result-x2)/result << std::endl;

  
  float x3=0;
  for(int i=0; i<10000;++i) x3+=tenth;
  print(x3);
  float x4=0;
  for(int i=0; i<10000;++i) x4+=one;
  print(x4);

  return 0;

}




