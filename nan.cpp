#include <iostream> 
#include <cmath>

int main(int q, char**) {

 float v;
 if (q>2) 
   v = std::sqrt(float(q));
 else
   v = std::sqrt(-float(q));

 double pt = v;
 std::cout << pt << ' ' << v  << std::endl;
 if (pt == 0) std::cout << "zero pt " << pt << std::endl;
 if (v == 0) std::cout << "zero v " << v << std::endl;
 if (!(pt > 0)) std::cout << "pt  zero " << pt << std::endl;

};

