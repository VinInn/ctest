const int N=1024;
float a[1024];
float b[1024];
float c[1024];
float d[1024];


void select() {
  for (int i=0; i!=N; ++i) {
    float k = b[i];
   if (0!=a[i]) k=c[i]+d[i];
   b[i] = k;
  }
}


bool z[1024];
// not vectorized: control flow in loop
void ori() {
  for (int i=0; i!=N; ++i)
    z[i] = a[i]<b[i] && c[i]<d[i];
}

// not vectorized: no vectype for stmt: z[i_17] = D.2199_10;
// scalar_type: bool
void ori2() {
  for (int i=0; i!=N; ++i)
    z[i] = a[i]<b[i] & c[i]<d[i];
}


// not vectorized: control flow in loop.
int j[1024];
void foo1() {
  for (int i=0; i!=N; ++i)
    j[i] = a[i]<b[i] && c[i]<d[i];
}

// not vectorized: unsupported data-type bool
void foo2() {
  for (int i=0; i!=N; ++i)
    j[i] = int(a[i]<b[i]) & int(c[i]<d[i]);
}

// not vectorized: unsupported data-type bool
void foo3() {
  for (int i=0; i!=N; ++i)
    j[i] = int(a[i]<b[i]);
}

// not vectorized: unsupported data-type bool
void foo4() {
  for (int i=0; i!=N; ++i)
    j[i] = a[i]<b[i] ? 1L : 0L ;
}

void foo5() {
  for (int i=0; i!=N; ++i)
    j[i] = (a[i]<b[i] ? -1 : 0) & (c[i]<d[i] ? -1 : 0);
}

void foo51() {
  for (int i=0; i!=N; ++i)
    j[i] = (a[i]<b[i] ? -1 : 1) & (c[i]<d[i] ? -1 : 1);
}



unsigned char k[1024];
void foo6() {
  for (int i=0; i!=N; ++i)
    k[i] = (a[i]<b[i]) & (c[i]<d[i]);
}


void foo7() {
  for (int i=0; i!=N; ++i)                                                         
    k[i] = (a[i]<b[i] ? -1 : 0) & (c[i]<d[i] ? -1 : 0);
}



float r[1024];
void bar() {
  for (int i=0; i!=N; ++i)
    r[i] = (a[i]<b[i] ? 1.f : 0.f) *  ( c[i]<d[i] ? 1.f : 0.f);
}
