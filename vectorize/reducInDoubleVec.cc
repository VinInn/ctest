float x[1024];
float y[1024];

float q;

float go() {

  float res=0;

  for (unsigned int k=0; k<1000; ++k) {
    float res=0;
    for (unsigned int i=0; i<1024; ++i)
      res += x[i]*y[i]+x[15];
    x[15]+=res;
  }

  return res;
}
