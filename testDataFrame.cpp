#include <boost/bind.hpp>
#include <boost/iterator_adaptors.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include<vector>

class DataFrameContainer;

class DataFrame {
public: 
  
  typedef unsigned int id_type;
  typedef unsigned short data_type;


  inline
  DataFrame() : m_id(0), m_data(0){}
  inline
  DataFrame(id_type i, data_type const * idata) :
    m_id(i), m_data(idata){}
  
  inline
  DataFrame(DataFrameContainer const & icont,
	    size_t i);
  inline
  data_type operator[](size_t i) const {
    return m_data[i];
  }

  inline
  data_type & operator[](size_t i) {
    return data[i];
  }
  
  inline
  id_type id() const { return m_id;}
  
private:
  data_type * data() {
    return const_cast<data_type *>(m_data);
  }

  id_type m_id;
  data_type const * m_data;
  
};


/** an optitimized container that linearized a "vector of vector"
 * although it can be sorted internally it is strongly adviced to
 * fill it already sorted....
 */
class DataFrameContainer {
public:
  typedef unsigned int id_type;
  typedef unsigned short data_type;
  typedef std::vector<id_type> IdContainer;
  typedef std::vector<data_type> DataContainer;
  typedef std::vector<id_type>::iterator IdIter;
  typedef std::vector<data_type>::iterator DataIter;
  typedef std::pair<IdIter,DataIter> IterPair;
  typedef std::vector<id_type>::const_iterator const_IdIter;
  typedef std::vector<data_type>::const_iterator const_DataIter;
  typedef std::pair<const_IdIter,const_DataIter> const_IterPair;

  struct IterHelp {
    typedef DataFrame result_type;
    IterHelp(DataFrameContainer const & iv) : v(iv){}
    
    DataFrame operator()(int i) const {
      return DataFrame(v,i);
    } 
    private:
    DataFrameContainer const & v;
  };
  typedef boost::transform_iterator<IterHelp,boost::counting_iterator<int> > const_iterator;


  DataFrameContainer(){}

  explicit DataFrameContainer(size_t istride, size_t isize=0) :
    m_stride(istride),
    m_ids(isize), m_data(isize*m_stride){}
  
  void swap(DataFrameContainer & rh) {
    std::swap(m_stride,rh.m_stride);
    std::swap(m_ids,rh.m_ids);
    std::swap(m_data,rh.m_data);
  }

  void swap(IdContainer & iic, DataContainer & idc) {
    std::swap(m_ids,iic);
    std::swap(m_data,idc);
  }

  
  void resize(size_t isize) {
    m_ids.resize(isize);
    m_data.resize(isize*m_stride);
  }
  
  void push_back(id_type iid, data_type const * idata) {
    m_ids.push_back(iid);
    size_t cs = m_data.size();
    m_data.resize(m_data.size()+m_stride);
    std::copy(idata,idata+m_stride,m_data.begin()+cs);
  }

  IterPair operator[](size_t i) {
    return IterPair(m_ids.begin()+i,m_data.begin()+i*m_stride);
  }

  const_IterPair operator[](size_t i) const {
    return const_IterPair(m_ids.begin()+i,m_data.begin()+i*m_stride);
  }
  
  const_iterator begin() const {
    return  boost::make_transform_iterator(boost::counting_iterator<int>(0),
					   IterHelp(*this));
  }
  const_iterator end() const {
    return  boost::make_transform_iterator(boost::counting_iterator<int>(size()),
					   IterHelp(*this));
  }

  size_t stride() const { return m_stride; }

  size_t size() const { return m_ids.size();}

  data_type operator()(size_t cell, size_t frame) const {
    return m_data[cell*m_stride+frame];
  }
  
  data_type const * frame(size_t cell) const {
    return &m_data[cell*m_stride];
  }
  
  id_type id(size_t cell) const {
    return m_ids[cell];
  }
  
  // IdContainer const & ids() const { return m_ids;}
  // DataContainer const & data() const { return  m_data;}
  
private:
  // can be a enumerator, or a template argument
  size_t m_stride;

  IdContainer m_ids;
  DataContainer m_data;
  
};

#include <boost/bind.hpp>
#include <iterator>
#include <algorithm>
#include <iostream>

inline
DataFrame::DataFrame(DataFrameContainer const & icont,
		     size_t i) :
  m_id(icont.id(i)), m_data(icont.frame(i)){}



int main() {
  DataFrameContainer cont(10,10);

  for ( DataFrameContainer::const_iterator p=cont.begin(); p!=cont.end();p++)
    std::cout << (*p).id() << " ";

  //  std::copy(boost::make_transform_iterator(cont.begin(),boost::bind(&DataFrame::id,_1)),
  //	    boost::make_transform_iterator(cont.end(),boost::bind(&DataFrame::id,_1)),
  //	     std::ostream_iterator<int>(std::cout," "));
    std::cout << std::endl;
    
}
