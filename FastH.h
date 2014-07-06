#include<limits>


template<typename T> 
struct FastH {

  static unsigned int max() {
    std::numeric_limits<T>::max();
  }

  static bool fill(void * data, int bin) {
    return ++(static_cast<T*>(data)[bin]) ==  std::numeric_limits<T>::max();
  }
  
  static unsigned int data(void * data, int bin) {
    return static_cast<T*>(data)[bin];
  }
  
  static void copy(void * data, int nbin, unsigned short * out) {
    for (int i=0; i!=nbin; ++i) 
      out[i]=(static_cast<T const *>(data)[i]);
  }
  
  static void copy(void * data, int nbin, unsigned int * out) {
    for (int i=0; i!=nbin; ++i) 
      out[i]=(static_cast<T const *>(data)[i]);
  }

  template<typename V>
  static void * move(void * data, int nbin) {
    V * out = new  V[nbin];
    T * old = static_cast<T*>(data);
    for (int i=0; i!=nbin; ++i) 
      out[i]=old[i];
    delete [] old; 
    return out;
  }
  


};



struct DynH {
  void * data;
  float low;
  float invW;
  unsigned short nbin;
  char type;
  

  DynH(unsigned short n, float imin, float imax) : data(0), low(imin), invW(float(n)/(imax-imin)), nbin(n), type('0'){}
  
  ~DynH() {
    switch (type) {
    case 'c':
      delete [] static_cast<unsigned char *>(data);
      break;
    case 's':
      delete [] static_cast<unsigned short *>(data);
      break;
    case 'i':
      delete [] static_cast<unsigned int *>(data);
      break;
    default:
      break;
    }
  }


  void fill(float d) {
    float bin = (d-low)*invW;
    int ibin=bin;
    if (bin<0) ibin=nbin; 
    else if (bin>=nbin)  ibin=nbin+1;


    switch (type) {
    case '0' :
      {
	unsigned char * nw =  new unsigned char[nbin+2];
	for (int i=0; i!=nbin+2; ++i) 
	  nw[i] = 0;
	++nw[ibin];
	data = nw;
	type = 'c';
      }
      break;
    case 'c':
      if (FastH<unsigned char>::fill(data,ibin)) {
	data = FastH<unsigned char>::move<unsigned short>(data,nbin+2);
	type = 's';
      }
      break;
    case 's':
      if (FastH<unsigned short>::fill(data,ibin)) {
	data = FastH<unsigned short>::move<unsigned int>(data,nbin+2);
	type = 'i';
      }
      break;
    default:
      FastH<unsigned int>::fill(data,ibin);   
    }
  }
  
  void copy(unsigned int * out) const {
    switch (type) {
    case '0' :
      for (int i=0; i!=nbin+2; ++i) 
	out[i] = 0;
      break;
    case 'c':
      FastH<unsigned char>::copy(data,nbin+2,out);
      break;
    case 's':
      FastH<unsigned short>::copy(data,nbin+2,out);
      break;
    default:
      FastH<unsigned int>::copy(data,nbin+2,out);
      break;
    }
  }

};



/*
template<int N>
struct DynHbase {
  float low;
  float invW;

  DynHbase(float imin, float imax) : low(imin), invW(float(N)/(imax-imim)){}

  virtual ~dynHbase(){}
  virtual long long fill(float d)=0;
  virtual long long max() const=0;
  virtual long long  operator[](int i) const = 0;


};


template<int N>
struct EmptyH : DynHbase {

  EmptyH(float imin, float imax) : dynHbase<N>(imin,imax){}
  virtual ~EmptyH(){}
  virtual long long fill(float d){}
  virtual long long max() const { return 0;}
  virtual long long  operator[](int i) { return 0;}

}

template<typename T, int N>
struct FastH : public DynHbase<N> {
  
  T data[N+2];

  FastH(float imin, float imax) : dynHbase<N>(imin,imax){
    for (int i=0; i!=N+2; ++i) 
      data[i] = 0;
  }
  FastH(EmptyH<N> const & rhs) : dynHbase<N>(rhs) {
    for (int i=0; i!=N+2; ++i) 
      data[i] = 0;
  }

  FastH( dynHbase<N> const & rhs) : dynHbase<N>(rhs) {
    for (int i=0; i!=N+2; ++i) 
      data[i] = rhs[i];
  }

  template<typename V>
  FastH(FastH<V,N> const & rhs) : dynHbase<N>(rhs) {
    for (int i=0; i!=N+2; ++i) 
      data[i] = rhs.data[i];
  }
  
  template<<typename V>
  void copy( V * d) const {
    for (int i=0; i!=N+2; ++i) d[i]=data[i];
  }

  long long fill(float d) {
    int ibin = (d-low)*invW;
    if (ibin<0) ibin=N; 
    else if (ibin>=N)  ibin=N+1;
    return data[ibin]++;
  }
  
  long long operator[](int i) const {return data[i];}

  long long max() const {
    numerical_limits<T>::max();
  }

};



template<int N>
struct dynH {
  enum type { empty, micro, mini, norm, huge}; 

  dynH(float imin, float imax) : h(new EmptyH<N>(imin,imax)),  htype(empty) {}
  ~dynH(){ delete h;}

  void fill(float d) {
    if (empty==htype) { dynHbase * old= h; h = new FastH<unsigned char, N>(*old); htype=micro; delete old;}

    long long ret = (*h).fill(d);
    if ((*h).max()==ret)  { 
      dynHbase * old= h; 

      if (micro==htype) {
	h = new FastH<unsigned short, N>(*old);
	htype=mini;
      }
      else if (mini====htype) {
	h = new FastH<unsigned int, N>(*old);
	htype=norm;
      }
      else if (norm====htype) {
	h = new FastH<unsigned long long, N>(*old);
	htype=huge;
      }

      delete old; 
    }
  }

  long long  operator[](int i) const {
    h ? (*h)[i] : 0;
  }


  dynHbase * h;
  type htype;
};

*/
