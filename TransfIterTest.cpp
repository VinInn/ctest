#include <boost/iterator_adaptors.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/bind.hpp>
#include <algorithm>

#include<vector>
#include<iostream>

struct M {

  float & operator()(int i,int j) {
    return m[i][j];
  }
  
  /*
  const float & operator()(int i,int j) const {
    return m[i][j];
  }

  */

  const float & data(std::pair<int,int> const& p) const {
    return m[p.first][p.second];
  }

  float m[3][3];
};


template<typename T>
const float & cg(const T& t,std::pair<int,int> const& p) {
  return  t(p.first,p.second);
}

template<typename T>
float & g(T& t,std::pair<int,int> const& p) {
  return  t(p.first,p.second);
}

int main() {

  M m; 
  float k[3][3] = { {1.,2.,3.},  {4.,5.,6.},  {7.,8.,9.} };
  std::copy(&k[0][0],&k[3][3],&m.m[0][0]);

  typedef std::vector<std::pair<int,int> > VI;
  std::vector<std::pair<int,int> > v;
  v.push_back(std::pair<int,int>(2,2));
  v.push_back(std::pair<int,int>(1,1));
  v.push_back(std::pair<int,int>(0,0));

  // external function
  {

    typedef
      boost::_bi::bind_t<float&, float& (*)(M&, const std::pair<int, int>&), boost::_bi::list2<boost::reference_wrapper<M>, boost::arg<1> > > Function;

    typedef
      boost::_bi::bind_t<const float&, const float& (*)(const M&, const std::pair<int, int>&), boost::_bi::list2<boost::_bi::value<M*>, boost::arg<1> > > ConstFunction;

    // boost::_bi::bind_t<boost::_bi::unspecified, g<M>, typename boost::_bi::list_av_2<M, boost::arg<1> >::type> Function;
    typedef boost::transform_iterator<Function,VI::const_iterator, const float&> const_iterator;
    typedef boost::transform_iterator<Function,VI::const_iterator, float&> iterator;
    
    iterator b = 
      boost::make_transform_iterator(v.begin(),boost::bind(g<M>,boost::ref(m),_1));
    const_iterator p = 
      boost::make_transform_iterator(v.begin(),boost::bind(g<M>,boost::ref(m),_1));
    iterator e = 
      boost::make_transform_iterator(v.end(),boost::bind(g<M>,boost::ref(m),_1));
   
    // *b = 4;
 
    for (;p!=e;p++) {
      float f =*p;
      std::cout << f << std::endl;
    }

    std::cout << "m(0,0) " << m(0,0) << std::endl;

    // std::sort(b,e);
    std::sort(v.begin(),v.end(), boost::bind(std::less<float>(), boost::bind(g<M>,boost::ref(m),_1), boost::bind(g<M>,boost::ref(m),_2)));
    
    p=b;
    for (;p!=e;p++) {
      std::cout << *p << std::endl;
    }
    std::cout << "m(0,0) " << m(0,0) << std::endl;


  }
  // member function
  {

    typedef
      boost::_bi::bind_t<const float&, boost::_mfi::cmf1<const float&, M, const std::pair<int, int>&>, boost::_bi::list2<boost::_bi::value<M*>, boost::arg<1> > > Function;

    typedef boost::transform_iterator<Function,VI::const_iterator> const_iterator;
    
    const_iterator p = 
      boost::make_transform_iterator(v.begin(),boost::bind(&M::data,&m,_1));
    const_iterator e = 
      boost::make_transform_iterator(v.end(),boost::bind(&M::data,&m,_1));
    
    for (;p!=e;p++)
      std::cout << (*p) << std::endl;

    

  }

  

}
