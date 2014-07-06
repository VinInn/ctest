#include<iostream>
#include <vector>


void part(unsigned int N, unsigned int bsize) {

  auto nb = N/bsize + ((0==N%bsize) ? 0 : 1);

  auto kb=nb/2; 
  auto ib = nb-kb;
  std::vector<unsigned int> blocks;
  blocks.push_back(0);
  while (kb>0) {
    blocks.push_back(blocks.back()+ib);
    ib = kb;
    kb /= 2; // ok >>1
    ib -=kb;
  }
  std::cout << N << " " << nb << ": " << blocks.size();
  for ( auto b : blocks) std::cout << ' ' << b;
  std::cout << std::endl;
  for ( auto b : blocks) std::cout << ' ' << b*bsize;
  std::cout << " ,  " << N - bsize*blocks.back()<< std::endl;

}



int main() {

  part(40000,512);
  part(400000,512);
  part(400,512);
  part(20*512,512);
  part(32*512,512);


 return 0;
}
