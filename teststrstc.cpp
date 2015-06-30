#include<iostream>
#include<sstream>


int main() {

  const char * bla = "5.6";
  const char * bla2 = "3";

 std::stringstream ss;
 ss <<bla;
  
 float f;
 int i;
 ss >> f;

 std::cout << f << std::endl;

 ss.str("");
 ss.clear();
 ss.str(bla2);
 // ss << bla2;
 ss >> i;

 std::cout << i << std::endl;



 return 0;

}
