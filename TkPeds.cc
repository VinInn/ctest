#include<iostream>
#include<sstream>
#include<vector>
#include<iomanip>

struct TkPed1 {
  bool disable      : 1;
  unsigned int low  : 6;
  unsigned int high : 6;
  unsigned int noise: 9;
  unsigned int ped  : 10;
};

struct TkPed2 {
  unsigned int ped  : 10;
  unsigned int noise: 9;
  unsigned int high : 6;
  unsigned int low  : 6;
  bool disable      : 1;
};


#include<cstdio>
int main(int clineN, char * clineM[]) {

  typedef std::vector<unsigned char> PedData;
  PedData pedData;

  if (clineN<2) std::cout << "please provide ped string" << std::endl; 

  size_t len = std::strlen(clineM[1]);
  std::istringstream in(clineM[1]);
  char c[9]; c[8] = '\0';
  // read as int...
  while (in) {
    in >> std::setw(sizeof(c)) >> c;
    pedData.resize(pedData.size()+4);
    unsigned int crap = ::strtoul(c,0,16);
    ::memcpy((void*)(&pedData[pedData.size()-4]),(void*)(&crap),4);
    // test
    for (size_t j=0;j<4;j++)
      std::cout << std::hex << std::setfill('0') << std::setw(2) << (int)(pedData[pedData.size()-4+j]);
  }
  std::cout << std::endl;
  
  // read in (as in reco)

  for (size_t j=0;j<pedData.size();j+=4) {
    const TkPed1 & tkped = (const TkPed1&)(pedData[j]);
    std::cout << tkped.ped << "," << tkped.noise << " ";
  }
  std::cout << std::endl;
  for (size_t j=0;j<pedData.size();j+=4) {
    const TkPed2 & tkped = (const TkPed2&)(pedData[j]);
    std::cout << tkped.ped << "," << tkped.noise << " ";
  }
  std::cout << std::endl;


  return 0;
}
