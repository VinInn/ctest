struct V {

 double * begin() { return v; }
 double * end()   { begin()+n;}

 double & operator[](int i) { return *(begin()+i);}

 double * v;
 int n;
};

struct VA {

 double * begin() { double * __restrict__ x = (double*)__builtin_assume_aligned(v, 32); return x; }
 double * end()   { begin()+n;}
 double & operator[](int i) { return *(begin()+i);}

 double * v;
 int n; 
};



void foo(V & a, V &b, V & c) {

  int i=0;for(auto & q : a) {
    q = b[i]+c[i];i++;
 }


}


void bar(VA & a, VA &b, VA & c) {

  int i=0;for(auto & q : a) {
    q = b[i]+c[i];i++;
 }


}

