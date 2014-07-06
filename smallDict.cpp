#ifndef condSmallWORMDict
#define condSmallWORMDict

#include<vector>
#include<string>
#include<algorithm>
#include<numeric>
#include<exception>
#include <boost/bind.hpp>
#include <boost/iterator_adaptors.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>


#define private public

namespace cond {

/** A small WORM Dictionary of small words
    optimized to do a single allocation
 */

  class SmallWORMDict {
  public:
    SmallWORMDict();
    ~SmallWORMDict();
    
    struct Frame {
      Frame(){}
      Frame(char const * ib,
	    unsigned int il,
	    unsigned int iind) :
	b(ib),l(il),ind(iind){}
      char const * b;
      unsigned int l;
      unsigned int ind;
    };

    struct IterHelp {
      typedef Frame result_type;
      IterHelp() : v(0){}
      IterHelp(SmallWORMDict const & iv) : v(&iv){}
      
      result_type const & operator()(int i) const {
	int k = (0==i) ? 0 : v->m_index[i-1]; 
	return  frame(&v->m_data[k], v->m_index[i]-k, i);
      } 
      
      Frame const & frame(char const * b,
			  unsigned int l,
			  unsigned int ind) const { 
	f.b = b; f.l=l; f.ind=ind;
	return f;
      }
      
    private:
      SmallWORMDict const * v;
      mutable Frame f;
    };
    
    friend struct IterHelp;

    typedef boost::transform_iterator<IterHelp,boost::counting_iterator<int> > const_iterator;

 
    const_iterator begin() const {
      return  boost::make_transform_iterator(boost::counting_iterator<int>(0),
					     IterHelp(*this));
    }
    
    const_iterator end() const {
      return  boost::make_transform_iterator(boost::counting_iterator<int>(size()),
					     IterHelp(*this));
    }

    Frame operator[](int i) const {
      int k = (0==i) ? 0 : m_index[i-1]; 
      return  Frame(&m_data[k], m_index[i]-k, i);
      } 

    const_iterator find(std::string const & s) const;

    const_iterator find(char const * s) const;

    // constructror from config
    explicit SmallWORMDict(std::vector<std::string> const & idict);
    
    // find location of a word
    size_t index(std::string const & s) const;

    size_t index(char const * s) const;

    size_t size() const;

  private: 
    std::vector<char> m_data;
    std::vector<unsigned int> m_index;
  };


}

#endif

namespace cond {

  SmallWORMDict::SmallWORMDict(){}
  SmallWORMDict::~SmallWORMDict(){}

  SmallWORMDict::SmallWORMDict(std::vector<std::string> const & idict) :
    m_data(std::accumulate(idict.begin(),idict.end(),0,
			   boost::bind(std::plus<int>(),_1,boost::bind(&std::string::size,_2)))),
    m_index(idict.size(),1) {
    
    // sort (use index)
    m_index[0]=0; std::partial_sum(m_index.begin(),m_index.end(),m_index.begin());
    std::sort(m_index.begin(),m_index.end(), 
	      boost::bind(std::less<std::string>(),
			  boost::bind<const std::string&>(&std::vector<std::string>::operator[],boost::ref(idict),_1),
			  boost::bind<const std::string&>(&std::vector<std::string>::operator[],boost::ref(idict),_2)
			  )
	      );

    //copy
    std::vector<char>::iterator p= m_data.begin();
    for (size_t j=0; j<m_index.size(); j++) {
      size_t i = m_index[j];
      p=std::copy(idict[i].begin(),idict[i].end(),p);
      m_index[j]=p-m_data.begin();
    }

  }

  
  struct LessFrame {
    bool operator()(SmallWORMDict::Frame const & rh,SmallWORMDict::Frame const & lh) const {
	return std::lexicographical_compare(rh.b,rh.b+rh.l,lh.b,lh.b+lh.l);
    }
    
  };

  size_t SmallWORMDict::index(std::string const & s) const {
    return (*find(s)).ind;
  }
 
  size_t SmallWORMDict::index(char const * s) const {
    return (*find(s)).ind;
  }
    
  SmallWORMDict::const_iterator SmallWORMDict::find(std::string const & s) const {
    Frame sp(&s[0], s.size(),0);
    return 
      std::lower_bound(begin(),end(),sp, LessFrame());
  }

  SmallWORMDict::const_iterator SmallWORMDict::find(char const * s) const {
    Frame sp(s, ::strlen(s),0);
    return 
      std::lower_bound(begin(),end(),sp, LessFrame());
  }


  size_t SmallWORMDict::size() const { return m_index.size(); }



}

#include <iostream>

int main() {

  std::vector<std::string> dict;
  size_t tot=0;
  dict.push_back("Sneezy");
  tot+=dict.back().size();
  dict.push_back("Sleepy");
  tot+=dict.back().size();
  dict.push_back("Dopey");
  tot+=dict.back().size();
  dict.push_back("Doc");
  tot+=dict.back().size();
  dict.push_back("Happy");
  tot+=dict.back().size();
  dict.push_back("Bashful");
  tot+=dict.back().size();
  dict.push_back("Grumpy");
  tot+=dict.back().size();
  
  cond::SmallWORMDict  worm(dict);


  std::cout << dict.size() << " " << worm.m_index.size() << std::endl;
  std::cout << tot << " " << worm.m_data.size() << std::endl;
  std::copy(worm.m_index.begin(),worm.m_index.end(),
	    std::ostream_iterator<int>(std::cout," "));
  std::cout << std::endl;
  std::copy(worm.m_data.begin(),worm.m_data.end(),
	    std::ostream_iterator<char>(std::cout,""));
  std::cout << std::endl;

  cond::SmallWORMDict::Frame f = worm[2];
  std::copy(f.b,f.b+f.l,
	    std::ostream_iterator<char>(std::cout,""));
  std::cout << std::endl;
  
  int i = worm.index("Doc");
  f = worm[i];
  std::copy(f.b,f.b+f.l,
	    std::ostream_iterator<char>(std::cout,""));
  std::cout << " at " << i<< std::endl;

  f = *worm.find("Sleepy");
  std::copy(f.b,f.b+f.l,
	    std::ostream_iterator<char>(std::cout,""));
  std::cout << " at " << f.ind << std::endl;

return 0;

}
