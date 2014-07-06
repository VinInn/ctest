#include <string>
#include <iostream>
#include <algorithm>
#include <set>
/////////////////
bool IsLightIon(const std::string & n)
{
  static const std::string snames[] = { "proton", "neutron", "alpha", "deuteron",
                           "triton", "He3", "GenericIon"};
  //  static const std::set<std::string> names(snames, snames+7); 
   // return true if the particle is pre-defined ion
  return std::find(snames, snames+7, n)!=snames+7;
  // return names.find(n)!=names.end();
} 

int main() {

  static const std::string names[] = { "proton", "neutron", "alpha", "deuteron",
				       "triton", "He3", "GenericIon"};
  for (int i=0;i<7;i++)
    if (!IsLightIon(names[i])) std:: cout <<"error for " << names[i] << std::endl;

  static const std::string bad[] = { "prot", "iron", "quark"};

  for (int i=0;i<3;i++)
    if (IsLightIon(bad[i])) std:: cout <<"error for " << bad[i] << std::endl;

  return 0;

} 
