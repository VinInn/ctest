#include<vector>
#include<string>
#include<algorithm>
#include<numeric>
#include<iostream>

inline std::string format_vstring(const std::vector<std::string>& v)  {
    std::string retVal;
    auto ss = accumulate(v.begin(),v.end(),2,[](int a,std::string const & s){return a+s.length()+2;});
    retVal.reserve(ss);

    std::cout << retVal.capacity() << std::endl;
    //retVal = std::accumulate(v.begin(),v.end(),retVal, [](std::string const & ret ,std::string const & s){return ret+ (ret.empty() ? "{ " : ", ")+s;}) + " }";
    for_each(v.begin(),v.end(),[&](std::string const & s){retVal += (retVal.empty() ? "{ " : ", ")+s;});
    retVal+= " }";
    std::cout << retVal.capacity() << std::endl;

    return retVal;
  }




int main() {

 std::vector<std::string> v;

 for (int i=0; i<10; ++i) 
    v.push_back("something");

 auto ret = format_vstring(v);

 std::cout << ret << std::endl;

 return ret.size();;

}
