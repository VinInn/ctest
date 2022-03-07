#include <iostream>
#include <fstream>
#include <string>

// definetively NOT thread safe
struct perfStartStop {
   static  std::ofstream & out() {
      static std::ofstream me("/tmp/perf_ctl.fifo");
      return me; 
   }

   static void start() {
     out() << "enable" << std::endl;
   } 

   static void stop() {
     out() << "disable" << std::endl;
   }
};
