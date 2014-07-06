#include<vector>
#include<algorithm>
#include<cmath>

struct A {
 A(float q=0): v(q){} 
 float v;
 bool operator<(A const & a) const { return v<a.v;}
};

struct B { 
 B(float q=0): v(q){}
 bool operator<(B const & a) const { return v<a.v;}

 float v;

};


struct C {
 C(double q=0): v(q){}
 bool operator<(C const & a) const { return v<a.v;}

 double v;

};



float cosq(A const & a) {
   return cos(a.v);
}
float cosq(B const & a) {
   return cos(a.v);
}



template<typename T> 
int game(std::vector<T> const & a, std::vector<T> & b) {
   typedef typename std::vector<T>::const_iterator Iter;
   for (Iter i=a.begin(); i!=a.end(); ++i) {
	if ( (*i).v>0.) b.push_back((*i).v+1);
   }
   std::sort(b.begin(),b.end());
   return b.size();
}

template<typename T> 
int gamep(std::vector<T*> const & a, std::vector<T*> & b) {
   typedef typename std::vector<T*>::const_iterator Iter;
   for (Iter i=a.begin(); i!=a.end(); ++i) {
        if ( (*i)->v>0.) b.push_back((*i));
   }
   std::sort(b.begin(),b.end());
   return b.size();
}

namespace data /* __attribute__((visibility("default"))) */ {

std::vector<A> a; 
std::vector<B> b;
std::vector<C> c;

std::vector<A*> ap;
std::vector<B*> bp;
std::vector<C*> cp;

}

#include<iostream>
int __attribute__((visibility("default"))) go() {
  using namespace data;
  int ret=0;	
try {
  ret+=game(a,a);
  ret+=game(b,b);
  ret+=game(c,c);

  ret+=gamep(ap,ap);
  ret+=gamep(bp,bp);
  ret+=gamep(cp,cp);
} catch (std::exception & ce) {
  std::cout << ce.what() << std::endl;
}

  return ret;

}



