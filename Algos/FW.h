#ifndef FW_H
#define FW_H
#include<vector>

struct Ev {
  int get(int) const; 
};

struct Es {
  int get(int) const; 
};


using Config = std::vector<int>;


#endif
