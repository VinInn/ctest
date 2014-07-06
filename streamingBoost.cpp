#include <fstream>
#include <iostream>

// include headers that implement a archive in simple text format
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>

void writeIt() {

  float f=3;
  int i=2;
  std::string s="Hello W";
  std::vector<double> v(3);
  v[0]=1;v[1]=2;v[2]=3;

  std::ofstream of("aTest.str");

  boost::archive::binary_oarchive ar(of);
  ar << f;
  ar << i;
  ar << s;
  ar << v;


}

void readIt() {

  float f;
  int i;
  std::string s;
  std::vector<double> v;
 

  std::ifstream fs("aTest.str");

  boost::archive::binary_iarchive ar(fs);
  ar >> f;
  ar >> i;
  ar >> s;
  ar >> v;

  std::cout << f
	    << " " << i
	    << " |" << s << "|"
	    << " " << v.size()
	    << " (" << v[0] <<","<<v[1]<<","<<v[2]<<")"
	    << std::endl;

}


int main(int argc, char * argv[] ) {

  if (argc>1) 
    writeIt();
 
  readIt();
  return 0;
}
