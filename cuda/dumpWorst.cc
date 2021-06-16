#include<iostream>
#include<fstream>
#include<cstdlib>
#include<cassert>

int main(int argc, char ** argv) {

   double worst[30*8];

   std::ifstream ifile; 
   ifile.open(argv[1],std::ios::binary);
   assert(ifile.good());
 
   ifile.read((char*)worst, sizeof(worst));
   assert(ifile.good());
   ifile.close();

   for (auto val : worst) 
     printf("%a\n",val);


  return 0;
}
