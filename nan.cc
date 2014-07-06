#include <cmath>

void a();

void nan(int q, char**) {

 float v;
 if (q>2)
   v = std::sqrt(float(q));
 else
   v = std::sqrt(-float(q));

 double pt = v;
 if (pt == 0) a();

};

