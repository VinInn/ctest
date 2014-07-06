#ifndef CMSSW_mayown_ptr_H
#define CMSSW_mayown_ptr_H

#include<cassert>
#include<cstring>

// a smart pointer which may own 
// can be implemented trivially with shared_ptr
// this is infinetely ligher
// assume alignment > 2....

template<typename T, int N=sizeof(T*)>
class mayown_ptr {
private:
  T const * p=nullptr;

  void markOwn() {
    unsigned char buff[N]; memcpy(buff,&p,N);
    assert((buff[N-1]&1)==0);
    ++buff[N-1];
    memcpy(&p,buff,N);
  }
  bool isOwn() const {
    unsigned char buff[N]; memcpy(buff,&p,N);
    return 1==(buff[N-1]&1);
  }


  T const * pointer() const {
    unsigned char buff[N]; memcpy(buff,&p,N);
    buff[N-1] &= 0xFE;
    T const * np;
    memcpy(&np,buff,N);
    return np;
  }

  void destroy() {
    if (isOwn()) delete const_cast<T*>(pointer());
  }
public:
  mayown_ptr(T * ip) : p(ip) { markOwn();}
  mayown_ptr(T const & ip) : p(&ip) {}
  ~mayown_ptr() { destroy();}
  mayown_ptr(mayown_ptr &)=delete;
  mayown_ptr(mayown_ptr && rh) : p(rh.p) { rh.p=nullptr;} 
  mayown_ptr& operator=(mayown_ptr &)=delete;
  mayown_ptr& operator=(mayown_ptr && rh) { destroy(); p=rh.p; rh.p=nullptr; return *this;}

  T const & operator*() const { return *pointer();}
  T const * get() const { return pointer();}
  T const * release() { auto np=pointer(); p=nullptr; return np;}

  T const * raw() const { return p;}
};

template<typename T>
bool operator==(mayown_ptr<T> const & rh, mayown_ptr<T> const & lh) {
  return rh.raw() == lh.raw();
}  
template<typename T>
bool operator<(mayown_ptr<T> const & rh, mayown_ptr<T> const & lh) {
  return rh.raw() < lh.raw();
}  

#endif

#include<vector>
#include<iostream>
int main() {

  using PD=mayown_ptr<double>;
  using VD=std::vector<PD>;

  std::vector<double> dd(10,3.14);
  VD vd;
  for (int i=0; i<10; ++i) {
    if (i%2==0) vd.push_back(PD(dd[i]));
    else vd.push_back(PD(new double(-1.2)));
  }

  VD vd2; std::swap(vd,vd2);
  for (int i=0; i<10; ++i) std::cout << *vd2[i] << std::endl;

  return 0;
}
