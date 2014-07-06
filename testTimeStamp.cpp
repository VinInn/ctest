#include <ctime>
#include <sys/time.h>
#include <string>
#include <iostream>
#define BOOST_DATE_TIME_POSIX_TIME_STD_CONFIG
#include "boost/date_time/posix_time/posix_time.hpp"


typedef unsigned long long TimeValue_t;
const TimeValue_t kLowMask(0xFFFFFFFF);


::timeval to_timeval(TimeValue_t iValue) {
  ::timeval stv;
  //  stv.tv_sec =  static_cast<unsigned int>(iValue >> 32);
  //stv.tv_usec = static_cast<unsigned int>(kLowMask & iValue);
  stv.tv_sec =  iValue >> 32;
  stv.tv_usec = kLowMask & iValue;
  return stv;
}
 

 
int main() {

   {

  boost::posix_time::ptime bt0 = boost::posix_time::from_time_t(0);
  std::cout << bt0 << std::endl;
  boost::posix_time::ptime btF = boost::posix_time::from_time_t(0x7FFFFFFF);
  std::cout << std::numeric_limits<int>::max() << std::endl;
  std::cout << btF << std::endl;
  long long se = std::numeric_limits<long long>::max()/(long long)1000000000;
  long long su = (std::numeric_limits<long long>::max()/1000LL)%(long long)1000000;
  boost::posix_time::ptime btE = bt0 +  boost::posix_time::seconds(se) + boost::posix_time::microseconds(su);
  std::cout << se << " " << su << std::endl;

  std::cout << btE << std::endl;
  std::cout << std::endl;



  TimeValue_t time = 0LL;
  ::timeval stv;
  ::gettimeofday(&stv,0);
  time = stv.tv_sec;
  time = (time << 32) + stv.tv_usec;
  
  std::cout << stv.tv_sec << " " << stv.tv_usec << std::endl;
  std::cout << time << std::endl;
  stv = to_timeval(time);
  std::cout << stv.tv_sec << " " << stv.tv_usec << std::endl;


   boost::posix_time::ptime bt = bt0 + boost::posix_time::seconds(stv.tv_sec) + boost::posix_time::microseconds(stv.tv_usec);
   bt +=  boost::posix_time::nanoseconds(19*25);
   boost::posix_time::time_duration td = bt - bt0;

  std::cout << bt << std::endl;
  std::cout << "s. " << td.total_seconds()  << "." << td.fractional_seconds() << std::endl;
  std::cout << "us " << td.total_microseconds()  << std::endl;
  std::cout << "ns " << td.total_nanoseconds()  << std::endl;
  std::cout << std::endl;

  }

  {
  TimeValue_t time = 0LL;
  ::timeval stv;
  ::gettimeofday(&stv,0);
  time = stv.tv_sec;
  time = (time << 32) + stv.tv_usec;
  
  std::cout << stv.tv_sec << " " << stv.tv_usec << std::endl;
  std::cout << time << std::endl;
  stv = to_timeval(time);
  std::cout << stv.tv_sec << " " << stv.tv_usec << std::endl;
  
  }


  return 0;

}
