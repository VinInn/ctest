#include <cmath>
#include <iostream>
#include <boost/thread/thread.hpp>
#include <boost/ref.hpp>

#include "RealTime.h"

inline double xtime() {
/*  
  timespec req = {0,0};
  clockid_t cid = CLOCK_THREAD_CPUTIME_ID;
  // pthread_getcpuclockid(pthread_self(), &cid);
  ::clock_gettime(cid,&req);
  return double(req.tv_sec)+1.e-9*double(req.tv_nsec);
  */
  return std::clock();
}


 struct F1 {
   double r;
   double e;
   void operator()()
   {
     r=0;
 //    double s = std::clock();
     double s = xtime();
     for (double i=0; i<40000000; i++)
       r += pow(i,2);
     e = xtime()-s;
   }
   void print() const
   {
     std::cout << "pow2 " << e << std::endl;
   }
 };

struct F2 {
  double r;
  double e;
  void operator()()
  {
    r=0;
    double s = xtime();
    for (double i=0; i<40000000; i++)
      r += std::pow(i,2);
    e = xtime()-s;
   }
   void print() const
   {
    std::cout << "std::pow2 " << e << std::endl;
  }
};

struct F3 {
  double r;
  double e;
  void operator()()
  {
    r=0;
    double s = xtime();
    for (double i=0; i<40000000; i++)
      r += i*i;
    e = xtime()-s;
   }
   void print() const
   {
    std::cout << "* " << e << std::endl;
  }
};

struct F4 {
  double r;
  double e;
  void operator()()
  {
    r=0;
    double s = xtime();
    for (double i=0; i<40000000; i++)
      r += pow(i,4);   
    e = xtime()-s;
   }
   void print() const
   {
    std::cout << "pow4 " << e << std::endl;
  }
};

struct F5 {
  double r;
  double e;
  void operator()()
  {
    r=0;
    double s = xtime();
    for (double i=0; i<40000000; i++)
      r += std::pow(i,4);
    e = xtime()-s;
  }
  void print() const
  {
    std::cout << "std::pow4 " << e << std::endl;
  }
};

struct F6 {
  double r;
  double e;
  void operator()()
  {
    r=0;
    double s = xtime();
    for (double i=0; i<40000000; i++)
      r += i*i*i*i;
    e = xtime()-s;
  }
  void print() const
  {
    std::cout << "*** " << e << std::endl;
  }
};

int main(int argc) {
  
  perftools::TimeType start = perftools::realTime();
  
  F1 f1; F2 f2;  F3 f3;  F4 f4;  F5 f5;  F6 f6;

  if (argc>1) {
    boost::thread t1(boost::ref(f1)); boost::thread t2(boost::ref(f2)); boost::thread t3(boost::ref(f3));
    boost::thread t4(boost::ref(f4)); boost::thread t5(boost::ref(f5)); boost::thread t6(boost::ref(f6));
    t1.join(); t2.join(); t3.join(); t4.join(); t5.join(); t6.join(); 
  }else {
    f1(); f2(); f3(); f4();  f5(); f6();
  }

  f1.print(); f2.print(); f3.print(); f4.print();  f5.print(); f6.print();

  perftools::TimeType end = perftools::realTime();
  
  std::cout << "tot real time " << 1.e-9*(end-start) << std::endl;
  
  return 0;
  
}

