#include <iostream>
#include <iostream>
#include <set>
int main() {

  std::cout << "hello Vin" << std::endl;

  std::set<std::string> all;
  all.insert("MU_");
  all.insert("MUa");
  all.insert("MU$");
  all.insert("MU*");
  all.insert("MU1");
  all.insert("MU1");
  all.insert("MU{");
  all.insert("MUz");
  all.insert("MU?");
  all.insert("MU,");
  all.insert("MU/");
  all.insert("MUandme");
  all.insert("MUzzz__12");

  char c('z');
  c++;

  std::cout << c << std::endl;

  for (std::set<std::string>::const_iterator p=all.begin();p!=all.end();p++)
    std::cout << *p << std::endl;



  return 0;


}
