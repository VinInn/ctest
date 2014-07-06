#include<memory>


namespace extstd {

  template<typename T>
  struct clone_ptr : public std::unique_ptr<T> {
    
    template<typename... Args>
    explicit clone_ptr(Args&& ... args)  noexcept : std::unique_ptr<T>(std::forward(args)...){}
    
    clone_ptr(clone_ptr const & rh) : std::unique_ptr<T>(rh? rh->clone() : nullptr){}
    clone_ptr(clone_ptr && rh) noexcept : std::unique_ptr<T>(std::move(rh)) {}
    
    clone_ptr & operator=(clone_ptr const & rh) {
      if (&rh!=this) this->reset(rh? rh->clone() : nullptr);
      return *this;
    }
    clone_ptr & operator=(clone_ptr && rh) noexcept {
      if (&rh!=this) std::unique_ptr<T>::operator=(std::move(rh));
      return *this;
    }
    
    
    template<typename U>   
    clone_ptr(clone_ptr<U> const & rh) : std::unique_ptr<T>(rh ? rh->clone() : nullptr){}
    template<typename U>
    clone_ptr(clone_ptr<U> && rh)  noexcept : std::unique_ptr<T>(std::move(rh)) {}
    
    template<typename U>
    clone_ptr & operator=(clone_ptr<U> const & rh) {
      if (&rh!=this) this->reset(rh? rh->clone() : nullptr);
      return *this;
    }
    template<typename U>
    clone_ptr & operator=(clone_ptr<U> && rh) noexcept {
      if (&rh!=this) std::unique_ptr<T>::operator=(std::move(rh));
      return *this;
    }
    
  };

}


#include<iostream>
#include<algorithm>
#include<cassert>

int cla=0;

struct A{
   A(){}
   A(int j) : i(j){}

  A * clone() const { cla++; std::cout<< "c A " << i << std::endl; return new A(*this);}

  int i=3;
};


int da=0;
int da0=0;
struct B{

  B(){}
  B(B const &b) : a(b.a){}
  B(B &&b)   noexcept : a(std::move(b.a)){}

  B & operator=(B const &b) {
    a=b.a;
    return *this;
  }
  B & operator=(B&&b)  noexcept {
    a=std::move(b.a);
    return *this;
  }

  ~B() {if(a) da++; else da0++; std::cout<< "d B " << (a ? a->i : -99) << std::endl;}

  extstd::clone_ptr<A> a;
};


#include<vector>
int main() {

  B b; b.a.reset(new A(2));


  B c = b;
  assert(cla==1);
  B d = b;
  assert(cla==2);

  b.a.reset(new A(-2));

  std::cout<< c.a->i << std::endl;

  c = b;
  assert(cla==3);

  std::cout<< c.a->i << std::endl;
  c.a.reset(new A(-7));

  std::vector<B> vb(1); 
  vb.push_back(b);
  assert(cla==4);

  vb.push_back(std::move(c));
  vb[0]=d;
  assert(cla==5);
  assert(da==0);

  std::cout<< vb[0].a->i << std::endl;
  std::sort(vb.begin(),vb.end(),[](B const & rh, B const & lh){return rh.a->i<lh.a->i;});
  std::cout<< (*vb[0].a).i << std::endl;
  std::swap(b,d);
  assert(cla==5);
  assert(da==0);
  std::cout << std::endl;


  return 0;
}
