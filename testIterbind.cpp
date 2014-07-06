#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <vector>
#include <algorithm>
#include <boost/iterator_adaptors.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

struct T {


  double z() const { return m_z;}

  double m_z;

};


namespace helpersT {

  inline double z(T const * t) {
    return (*t).z();
  }

  struct H {
    typedef double result_type;
    H(std::vector<T*> const & iv) :v(iv){}

    double operator()(int i) const {
      return v[i]->z();
    }

    std::vector<T*> const & v;
  };

}

struct Set {
  Set(double ii=0) : i(ii){}
  double i;
  void operator()(T * & t) {
    t = new T;
    t->m_z=i++;
  }
};


template<typename I>
size_t index(I b, I e, float v) {
  I p= e;
  p  = std::lower_bound(b,e,v);
  if (*p==v) return  p-b;
  else return e-b;
}

#include<iostream>
#include<iterator>

int main() {

  std::vector<T*> a(100);
  double z=-100.;
  for (std::vector<T*>::iterator it=a.begin(); it!=a.end(); it+=10) {
    std::for_each(it,it+10,Set(z));
    z+=30;
  }
  a[0]->m_z=-1000;
  a[1]->m_z=-900;
  // a[21].m_z=35;
  a[22]->m_z=45;
  a.back()->m_z=1000;
  std::random_shuffle(a.begin(),a.end());


  // sort
  std::sort(a.begin(),a.end(), boost::bind(std::less<float>(),
					   boost::bind(&T::z,_1),
					   boost::bind(&T::z,_2)
					   )
	    );
  
  // look for a value...
  size_t i = index(boost::make_transform_iterator(a.begin(),helpersT::z),
  		   boost::make_transform_iterator(a.end(),helpersT::z),
		   15.);
  size_t j = index(boost::make_transform_iterator(a.begin(),helpersT::z),
  		   boost::make_transform_iterator(a.end(),helpersT::z),
		   27.);
  
  std::cout << i << " " << j  << std::endl;

  //  typedef boost::transform_iterator<,VI::const_iterator> const_iterator;


  std::copy(boost::make_transform_iterator(a.begin(),&helpersT::z),
	    boost::make_transform_iterator(a.end(),&helpersT::z),
	    std::ostream_iterator<float>(std::cout," "));
  std::cout << std::endl;

  boost::counting_iterator<int> b(0);
  boost::counting_iterator<int> e(a.size());
  typedef boost::transform_iterator<helpersT::H,boost::counting_iterator<int> > const_iterator;

  const_iterator bha =boost::make_transform_iterator(b,helpersT::H(a));
  std::copy(boost::make_transform_iterator(b,helpersT::H(a)),
	    boost::make_transform_iterator(e,helpersT::H(a)),
	    std::ostream_iterator<float>(std::cout," "));
  std::cout << std::endl;

  return 0;

}
