#include<utility>
template<typename RI, typename T>
constexpr RI lb(RI first, int len, T const & val) {
  return len>0 ? 
    ( *(first + (len>>1) ) < val ? lb(first + (len>>1)+1,len-(len>>1)-1,val) : lb(first,len>>1,val) )
    : first;
} 

template<typename RI, typename T>
constexpr RI ub(RI first, int len, T const & val) {
  return len>0 ? 
    ( val < *(first + (len>>1) )  ? ub(first,len>>1,val) : ub(first + (len>>1)+1,len-(len>>1)-1,val) )
    : first;
} 


template<typename RI, typename T>
constexpr std::pair<RI,RI> er(RI first, int len, T const & val) {
  return len>0 ? 
    ( *(first + (len>>1) ) < val ? er(first + (len>>1)+1,len-(len>>1)-1,val) :
      ( val < *(first + (len>>1) ) ? er(first,len>>1,val) :
	std::pair<RI,RI>(lb(first,len>>1,val),ub(first + (len>>1)+1,len-(len>>1)-1,val))
	)
      )
    : std::pair<RI,RI>(first,first);
} 


template<typename RI, typename T>
constexpr RI lb(RI first, RI last, T const & val) {
  return lb(first,last-first,val);
} 

template<typename RI, typename T>
constexpr RI ub(RI first, RI last, T const & val) {
  return ub(first,last-first,val);
} 

template<typename RI, typename T>
constexpr std::pair<RI,RI> er(RI first, RI last, T const & val) {
  return er(first,last-first,val);
} 

constexpr int one[]={1,1,1,1};
constexpr int seq[]={-4,-2,0,1,3,3,3,6,10};


bool foo() {

  return  lb(seq,9,2)-seq >3;

}

bool bar() {
  constexpr auto p = lb(seq,9,2);
  return p-seq > 3;

}

bool ext2() {
  constexpr auto p = er(seq,seq+9,2);
  return p.first==p.second;
}

bool ext3() {
  constexpr auto p = er(seq,seq+9,3);
  return p.first==p.second;
}



#include<algorithm>
#include <iostream>
int main() {

  std::cout << lb(one,one+4,1)-one << std::endl;
  std::cout << lb(one,one+4,0)-one << std::endl;
  std::cout << lb(one,one+4,2)-one << std::endl;

  for (int i=-5; i!=13; ++i)
    std::cout << i <<"," << lb(seq,seq+9,i)-seq << " ";
  std::cout << std::endl;
  for (int i=-5; i!=13; ++i)
    std::cout << i <<"," << ub(seq,9,i)-seq << " ";
  std::cout << std::endl;
  for (int i=-5; i!=13; ++i) {
    auto p = er(seq,9,i);
    std::cout << i <<"," << p.first-seq << ":" << p.second-seq << " ";
  }
  std::cout << std::endl;
  for (int i=-5; i!=13; ++i) {
    auto p = std::equal_range(seq,seq+9,i);
    std::cout << i <<"," << p.first-seq << ":" << p.second-seq << " ";
  }
  std::cout << std::endl;


}
