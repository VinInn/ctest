#include <vector>
#include <iostream>
#include<functional>
#include<algorithm>

typedef std::vector<double> VI;

struct V {
   VI v;

   V(){std::cout <<"dc" << std::endl;}

   virtual ~V(){std::cout <<"dd " << v.size()<< std::endl;}

   V(V const & rh) : v(rh.v) {
     std::cout <<"cc" << std::endl;
   }

   V(size_t n, double d) : v(n,d){} 

  size_t size() const { return v.size();}

   V & operator=(V const &rh) {
     std::cout <<"ac" << std::endl;
     V tmp(rh);
     v.swap(tmp.v);
     return *this;
  }

#if defined( __GXX_EXPERIMENTAL_CXX0X__)

   V(V && rh) : v(std::move(rh.v)) {
     std::cout <<"rc" << std::endl;
   }

   V & operator=(V && rh) {
     std::cout <<"ar" << std::endl;
     std::swap(rh.v,v);
     return *this;
  }

#else
  
  void swap(V & rh) {
    std::cout <<"V::swap" << std::endl;
    std::swap(rh.v,v);
  }


#endif

};

#if !defined( __GXX_EXPERIMENTAL_CXX0X__)
inline
void swap(V & lh, V & rh) {
    std::cout <<"::swap" << std::endl;
    std::swap(lh.v,rh.v);
}
#endif


inline bool operator<(V const& rh, V const& lh) {
  return rh.v[0]<lh.v[0];
}

struct IV : public V {
  IV(){}
  IV(size_t n, double d) : V(n,d){} 

};

struct CV {
  V v;
  CV(){}
  CV(size_t n, double d) : v(n,d){} 
  CV(V const & iv) : v(iv){} 

#if defined( __GXX_EXPERIMENTAL_CXX0X__)
  CV(V && iv) : v(std::forward<V>(iv)){} 

  /*
  CV(CV && rh) : v(std::forward<V>(rh.v)) {
   }

   CV & operator=(CV && rh) {
     v.operator=(std::forward<V>(rh.v));
     return *this;
  }
  */
#endif

};

inline bool operator<(CV const& rh, CV const& lh) {
  return rh.v<lh.v;
}


#include<typeinfo>

template<typename AV>
void one() {

  std::cout << "\n\nfor " << typeid(AV).name() << std::endl  << std::endl;
   std::vector<AV> vvs;
   std::cout << "push_back " << std::endl;
   vvs.push_back(AV(50,0.));
   for (int i=1;i<5; ++i) 
     vvs.push_back(AV(100*i,i));
   std::cout << "push_front " << std::endl;
   vvs.insert(vvs.begin(),AV(300,1.));
   

#if defined( __GXX_EXPERIMENTAL_CXX0X__)
   auto vov = std::bind(&AV::v,std::placeholders::_1);
   vov(vvs[0]).size();
#endif
   
//   auto myless = std::bind<bool>(std::less<VI>(),
//			   std::bind<VI const&>(&AV::v,std::placeholders::_1),
//			   std::bind<VI const&>(&AV::v,std::placeholders::_2)
//			   );

   //std::cout << ( myless(vvs[0],vvs[2]) ? "less" : "greater" ) << std::endl;


   std::cout << "shuffle " << std::endl;
   std::random_shuffle(vvs.begin(),vvs.end());
   
   std::cout << "sort " << std::endl;
   std::sort(vvs.begin(),vvs.end());

   //std::cout << "sort myless" << std::endl;
   //std::sort(vvs.begin(),vvs.end(), myless);
  


   {
     std::cout << "swap " << std::endl;
     AV v(5,3.);
     std::swap(v,vvs[3]);
     std::cout << "swap done" << std::endl;
    
   }


  std::cout << "the end" << std::endl << std::endl;
}




int main() {
  one<V>();
  one<IV>();
  one<CV>();


  V v(5,3.);
  std::cout << v.size() << std::endl;
  CV cv = v;
  std::cout << cv.v.size() << std::endl;
  std::cout << v.size() << std::endl;
  cv=V(7,3.);
  std::cout << cv.v.size() << std::endl;
  cv = v;
  std::cout << cv.v.size() << std::endl;
  std::cout << v.size() << std::endl;
  cv = CV(9,3.);
  std::cout << cv.v.size() << std::endl;
}
