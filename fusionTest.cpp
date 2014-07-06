#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>

#include <boost/fusion/sequence.hpp>

#include <boost/fusion/iterator/deref.hpp>

#include <boost/fusion/iterator/next.hpp>

#include <boost/fusion/iterator/prior.hpp>

#include <boost/fusion/iterator/equal_to.hpp>

#include <boost/fusion/iterator/distance.hpp>

#include <boost/fusion/iterator/advance.hpp>

#include <boost/fusion/iterator/value_of.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/sequence/generation/vector_tie.hpp>
#include <boost/bind.hpp>


template<typename Seq>
struct print_offset {
  print_offset(Seq const & iseq) : seq(iseq){}
  template<typename T>
  void operator()(T const & t) const {
    int off = (char*)(&t) - (char*)(&seq);
    std::cout << off << " " << t << std::endl;
  }
  Seq const & seq;
};

struct printIt {
  template <typename T, typename State>
  struct result {   typedef State type; };
  
  template<typename T>
  int  operator() (T const & t, int i) const {
    std::cout << "elem " << i 
	      <<" = " << t << std::endl;
    return ++i;
  }
};

struct Convert {
  template <typename T, typename State>
  struct result {   typedef State type; };
  
  template<typename T, typename It>
  It  operator() (T const & t, It i) const {
    convert(t,*i);
    return ++i;
  }
  template<typename T, typename V>
  void convert(T const & t, V & v)  const {
    v = t;
  }
  template<typename T>
  void convert(T const & t, std::string & v) const {
    static std::ostringstream os;
    os.str("");
    os << t;
    v = os.str();
  }
  //  std::ostringstream os;
};

int main() {

  typedef boost::fusion::vector<int, char, double, char const*> seq_type;
  seq_type t(1, 'x', 3.3, "hello");
  typedef boost::fusion::vector<int, char, double, long long> seq2;
  seq2 s(1, 'x', 3.3, 1234567890);
  boost::fusion::result_of::begin<seq_type>::type i(t);

  std::cout << *i << std::endl;

  int off = (char*)(&(*i)) - (char*)(&t);
  std::cout << off << std::endl;

  boost::fusion::for_each(t,print_offset<seq_type>(t));
  boost::fusion::fold(t,0,printIt());
  {
    std::vector<std::string> vs(boost::fusion::size(t));
    boost::fusion::fold(t,vs.begin(),Convert());
    std::copy(vs.begin(),vs.end(),std::ostream_iterator<std::string>(std::cout," "));
    std::cout << std::endl;
  }
  {
    std::vector<int> vs(boost::fusion::size(s));
    boost::fusion::fold(s,vs.begin(),Convert());
    std::copy(vs.begin(),vs.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout << std::endl;
  }

  return 0;

};
