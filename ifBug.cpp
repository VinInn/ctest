const int N=64;
float c[N];
float d[N];
int   j1[N];
int   j2[N];

void loop0() {
  for (int i=0; i!=N; ++i) {
    if (j1[i]<j2[i]) c[i] = -d[i];
  }
}


void loop1() {
  for (int i=0; i!=N; ++i) {
    if (c[i]<0) d[i] = -d[i];
  }
}

void loop2() {
  for (int i=0; i!=N; ++i) {
    float tmp = d[i];
    if (c[i]<0) tmp = -tmp;
    d[i]=tmp;
  }
}
