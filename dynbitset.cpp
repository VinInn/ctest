#include<boost/dynamic_bitset.hpp>
#include<iostream>
#include<algorithm>
#include<iterator>
#include<vector>

template<typename B,typename A>
void print(const  boost::dynamic_bitset<B,A>& a) {
  typedef std::vector<typename boost::dynamic_bitset<B,A>::block_type> Buffer;
  Buffer buf;
  boost::to_block_range(a,std::back_insert_iterator<Buffer>(buf));
  std::cout << std::hex;
  std::copy(buf.begin(),buf.end(),std::ostream_iterator<int>(std::cout, "."));
  std::cout << std::endl;
}

int main() {


  typedef boost::dynamic_bitset<> DBS;
  typedef boost::dynamic_bitset<>::block_type block_type; // (defalt is unsigned long...)

  std::cout << "bit per block " << DBS::bits_per_block << std::endl;


  block_type k0=0xef76;
  block_type k1=0x12345678;
  block_type k2=0xabcd;
  block_type k3=0x1234;
  DBS a; 
  a.append(k0);
  a.append(k1);
  DBS b; 
  b.append(k2);
  b.append(k3);

  print(a);
  print(b);


  std::cout << a << std::endl;
  std::cout << b << std::endl;

  typedef std::vector<block_type> Buffer;

  Buffer buf;

  boost::to_block_range(b,std::back_insert_iterator<Buffer>(buf));
  std::cout << std::hex;
  std::copy(buf.begin(),buf.end(),std::ostream_iterator<int>(std::cout, "."));
  std::cout << std::endl;
  a.append(buf.begin(),buf.end());
  std::cout << a << std::endl;

  print(a);


  return 0;

}
