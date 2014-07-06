#include<vector>
#include<map>
#include<sstream>
#include<algorithm>
#include<iterator>
#include<iostream>

struct Value {

  float a1;
  float a2;
  float a3;
  float a4;
  float a5;
  float a6;

};

bool operator==(const Value& a, const Value& b) {
  return ::memcmp(&a,&b,sizeof(Value))==0;
}

int main() {

  typedef std::map<int,Value> Map;

  Map m;
  for (int i=0;i<75000;i++) {
    Value v = {i,i,i,i,i,i};
    m.insert(std::make_pair(i,v));
    
  }

  std::ostringstream os;
  {
    std::vector<std::pair<Map::key_type,Map::mapped_type> > v(m.size());
    std::copy(m.begin(),m.end(),v.begin());
  
    os.write((char*)(&v.front()),m.size()*sizeof(Map::value_type));
  }

  {
    std::vector<std::pair<Map::key_type,Map::mapped_type> > v(os.str().size()/sizeof(Map::value_type));
    ::memcpy((char*)(&v.front()),&os.str()[0],os.str().size());

    Map m2;
    std::copy(v.begin(),v.end(),std::inserter(m2,m2.begin()));
    
    if (m==m2) std::cout << "ok" << std::endl;
    else std::cout << "Vincenzo, you failed" << std::endl;
  }



  return 0;
};
