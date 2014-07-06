float x[1024];
float y[1024];
float z[1024];
float w[1024];

int k[1024];
bool b[1024];

void barX() {
  for (int i=0; i<1024; ++i) {
    k[i] = (x[i]>0) & (w[i]<y[i]);
    z[i] = (k[i]) ? z[i] : y[i];
 }
}


void barB() {
  for (int i=0; i<1024; ++i) {
    b[i] = (x[i]>0) & (w[i]<y[i]);
    z[i] = (b[i]) ? z[i] : y[i];
 }
}


float u;
float q;

inline
bool in(float const & x, float const & y, float const & w) {
   auto v = x*u;
   return  (v>q) & (w<y) & (y>0.5f);
}

void bar() {
  for (int i=0; i<1024; ++i) {
    auto v = x[i]*u;
    auto c =  (v>q) & (w[i]<y[i]) & (y[i]>0.5f);
    z[i] = c ? y[i] : z[i];
 }
}

void bar2() {
  for (int i=0; i<1024; ++i) {
    auto c =  in(x[i],y[i],w[i]);
    z[i] = c ? y[i] : z[i];
 }
}




void barIf() {
  for (int i=0; i<1024; ++i) {
    auto c =  (x[i]>0) & (w[i]<y[i])  & (y[i]>0.5f);
    if (c) z[i] = y[i];
 }
}



