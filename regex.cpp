#include "regex.h"
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>


struct Regex {
  
  explicit Regex(const std::string & s) : m_ok(false), me(s) {
    size_t p = me.find(".");
    m_ok = p!=std::string::npos;
    if(m_ok) {
      if (p>0) {
	m_range.first = me.substr(0,p);
	m_range.second = m_range.first+"{"; // '{' is 'z'+1
      }
      me = "^" + me + "$";
      regcomp(&m_regex,me.c_str(),0);
    }
  } 
  
  ~Regex() { if(m_ok) regfree(&m_regex); }
  
  bool empty() const { return me.empty();}
  
  bool notRegex() const { return !m_ok;}
  
  const std::string & value() const { return me;}
  
  bool match(const std::string & s) const {
    if (m_ok)
      return !regexec(&m_regex, s.c_str(), 0,0,0);
    else
      return me==s;
  }
  
  const std::pair< std::string, std::string> & range() const { return m_range;}
private:
  bool m_ok;
  regex_t m_regex;
  std::string me;
  // range of me in a collating sequence
  std::pair<std::string, std::string> m_range;
};


int main() {
  
  std::set<std::string> dict;
  typedef std::set<std::string>::const_iterator DCI; 
  dict.insert("A"); 
  dict.insert("a");
  dict.insert("//MB");
  // dict.insert("//MB2P");
  dict.insert("//MB2P/MB2SuperLayerZ/MB2SLZLayer_58Cells/MB2SLZGas");
  dict.insert("//MB2P23P/MB2SuperLayerZ/MB2SLZLayer_58Cells/MB2SLZGas");
  dict.insert("//MB2P32P/MB2SuperLayerZ/MB2SLZLayer_58Cells/MB2SLZGas");
  dict.insert("//MB2P/MB2SuperLayerZ/MB2SLZLayer_59Cells/MB2SLZGas");
  dict.insert("//MB2P32P/MB2SuperLayerPhi/MB2SLPhiLayer_60Cells/MBSLPhiGas");
  dict.insert("//MB2P/MB2SuperLayerPhi/MB2SLPhiLayer_60Cells/MBSLPhiGas");

  DCI bn(dict.begin()), ed(dict.end());
  Regex aRegex("//MB2P.*");
  // Regex aRegex("//MB2P.*/MB2SuperLayerZ/MB2SLZLayer_.*Cells/MB2SLZGas");
  typedef std::vector<DCI> Candidates;
  Candidates candidates;
  if ( aRegex.notRegex() ) {
    DCI it = dict.find(aRegex.value());
    if (it!=ed) candidates.push_back(it);
  }
  else {
    if ( !aRegex.range().first.empty()) {
      bn =  dict.lower_bound(aRegex.range().first);
      ed =  dict.upper_bound(aRegex.range().second);
    }
    for (DCI it=bn; it != ed; ++it)
      if(aRegex.match(*it)) candidates.push_back(it);
  }
  for (int i=0; i<int(candidates.size()); i++)
    std::cout << *candidates[i] << std::endl;
  
  return 0;
}
