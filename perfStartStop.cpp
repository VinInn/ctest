#include <iostream>
#include <fstream>
#include <string>


#include "perfStartStop.h"
#include <cmath>



int main() {

   perfStartStop::start();   

   double k=0;
   for  (int i=0; i<1024*1024; ++i) k += sin(i*00001);

   perfStartStop::stop();
   return k<0;

};
