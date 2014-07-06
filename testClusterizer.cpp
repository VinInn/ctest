#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <vector>
#include <algorithm>
#include <boost/iterator_adaptors.hpp>
#include <boost/iterator/transform_iterator.hpp>


class SorterByFunction {
public:
  typedef float value_type;
  typedef std::vector<size_t> Indices; 
  typedef std::vector<value_type> Values;

  template<typename Iter,typename F> 
  void operator()(Iter b, Iter e, F f) {
    sort(b,e,f);
  }

  Indices const & indices() const {
    return m_indices;
  }

  // value_type value(size_t i) const { return m_values[i];}
  value_type sortedValue(size_t i) const { return m_values[m_indices[i]];}
 


  struct Gen {
    Gen(): n(0){}
    size_t operator()() { return n++;} 
    size_t n;
  };


private:
  
  template<typename Iter,typename F> 
  void sort(Iter b, Iter e, F f) {
    m_values.resize(e-b);
    std::transform(b,e,m_values.begin(),f);
    m_indices.resize(m_values.size());
    std::generate(m_indices.begin(),m_indices.end(),Gen());
    std::sort(m_indices.begin(),m_indices.end(),
	      boost::bind(std::less<float>(),
			  //			  boost::bind(&SorterByFunction::value,this,_1),
			  // boost::bind(&SorterByFunction::value,this,_2)
			  boost::bind<const float&>(&Values::operator[],boost::ref(m_values),_1),
			  boost::bind<const float&>(&Values::operator[],boost::ref(m_values),_2)
			  )
	      );
  }

private:

  Indices m_indices;
  Values m_values;

};


class Clusterize {
public:
  typedef SorterByFunction::value_type value_type;
  typedef SorterByFunction::Indices Indices; 
  typedef SorterByFunction::Values Values;

  explicit Clusterize(value_type minDist) :
    m_minDistance(minDist){}


  template<typename Iter,typename F> 
  void operator()(Iter b, Iter e, F f) {
    clusterize(b,e,f);
  }

  Indices const & clusters() const {
    return m_clusters;
  }

  Indices const & indices() const {
    return m_sorter.indices();
  }

private:

  struct DefineCluster {
    DefineCluster(Clusterize & result): 
      m_result(result), m_last(m_result.value(0)) {}

    void operator()(size_t i) {
      value_type v = m_result.value(i);
      if (v-m_last > m_result.m_minDistance)
	m_result.m_clusters.push_back(i);
      m_last=v;
    }

    Clusterize & m_result;
    value_type m_last;
  };

  friend struct DefineCluster;

  template<typename Iter,typename F> 
  void clusterize(Iter b, Iter e, F f) {
    m_sorter(b,e,f);
    DefineCluster df(*this);
    for(size_t i=0; i!=indices().size(); ++i) df(i);
    // close last interval....
    m_clusters.push_back(indices().size());
  }


private:

  value_type value(size_t i) const { return m_sorter.sortedValue(i);}
 

  
  value_type m_minDistance;
  SorterByFunction m_sorter;
  Indices m_clusters;
};

//---------------------------------------------------------

struct T {


  double z() const { return m_z;}

  double m_z;

};


namespace helpersT {

  /*
  inline double z(T const * t) {
    return (*t).z();
  }
  */
  
  inline double z(T const & t) {
    return t.z();
  }

}

struct Set {
  Set(double ii=0) : i(ii){}
  double i;
  void operator()(T& t) {
    t.m_z=i++;
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
  std::vector<T> a(100);
  double z=-100.;
  for (std::vector<T>::iterator it=a.begin(); it!=a.end(); it+=10) {
    std::for_each(it,it+10,Set(z));
    z+=30;
  }
  a[0].m_z=-1000;
  a[1].m_z=-900;
  // a[21].m_z=35;
  a[22].m_z=45;
  a.back().m_z=1000;
  std::random_shuffle(a.begin(),a.end());

  Clusterize clus(10.);
  clus(a.begin(),a.end(), boost::bind(&T::z,_1));

  std::cout << clus.clusters().size() << std::endl;
  std::copy(clus.clusters().begin(),clus.clusters().end(),
	    std::ostream_iterator<float>(std::cout," "));
  std::cout << std::endl;
  size_t first=0;
  for(Clusterize::Indices::const_iterator cl=clus.clusters().begin();
      cl!=clus.clusters().end();++cl) {
    for (size_t i=first; i!=*cl;++i) {
      std::cout << a[clus.indices()[i]].m_z << " ";
    }
    std::cout << std::endl;
    first=*cl;
  }

  // real sort
  std::sort(a.begin(),a.end(), boost::bind(std::less<float>(),
					   boost::bind(&T::z,_1),
					   boost::bind(&T::z,_2)
					   )
	    );

  // look for a value...

  //  std::lower_bound(boost::make_transform_iterator(a.begin(),boost::bind(&T::z,_1)),
  //		   boost::make_transform_iterator(a.end(),boost::bind(&T::z,_1)),
  //		   15.);

  
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

  return 0;

}
