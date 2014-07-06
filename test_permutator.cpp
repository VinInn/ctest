#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/permutation_iterator.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>


#include <vector>
#include <algorithm>
#include <numeric>
#include<iostream>

int main() {
  typedef  std::vector<double> Values;
  std::vector<double> a(100,1.);
  std::vector<double>::iterator it=a.begin();
  (*it)=-100.;
  for (;;) {
    it = std::partial_sum(it,it+10,it);
    if (it==a.end()) break;
    (*it) = *(it-1)+20;
  }
  std::copy (a.begin(),a.end(),std::ostream_iterator<double>(std::cout," "));
  std::cout << std::endl;
  a[0]=-1000;
  a[1]=-900;
  // a[21].m_z=35;
  a[22]=45;
  a.back()=1000;
  std::random_shuffle(a.begin(),a.end());

  std::vector<int> index(a.size(),1);
  index[0]=0;
  std::partial_sum(index.begin(),index.end(),index.begin());
  std::copy (index.begin(),index.end(),std::ostream_iterator<int>(std::cout," "));
  std::cout << std::endl;

  typedef boost::permutation_iterator< std::vector<double>::iterator, std::vector<int>::iterator > permutation_type;
  permutation_type begin = boost::make_permutation_iterator( a.begin(), index.begin() );
  permutation_type pit = begin;
  permutation_type end = boost::make_permutation_iterator( a.end(), index.end() );
  
  std::cout << "The permutated range is : ";
  std::copy( begin, end, std::ostream_iterator< double >( std::cout, " " ) );
  std::cout << std::endl;

  std::sort(index.begin(), index.end(),
	    boost::bind(std::less<double>(),
			boost::bind<const double&>(&Values::operator[],boost::ref(a),_1),
			boost::bind<const double&>(&Values::operator[],boost::ref(a),_2)
			)
	    );

  
  std::cout << "The sorted range is : ";
  std::copy( begin, end, std::ostream_iterator< double >( std::cout, " " ) );
  std::cout << std::endl;
 
  return 0;

};
