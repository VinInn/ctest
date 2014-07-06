#include<utility>
#include <string>
#include <limits>
namespace cond{  
  //typedef unsigned int Time_t;
  typedef unsigned long long Time_t;
  typedef std::pair<Time_t, Time_t> ValidityInterval;
  typedef enum { runnumber=0,timestamp,lumiid, userid } TimeType;
  const unsigned int TIMETYPE_LIST_MAX=4;
  const cond::TimeType timeTypeList[TIMETYPE_LIST_MAX]=
    {runnumber,timestamp,lumiid,userid};
  const std::string timeTypeNames[TIMETYPE_LIST_MAX]=
    {"runnumber","timestamp","lumiid","userid"};


  static const Time_t TIMELIMIT(0xFFFFFFFF);

  template<TimeType type>
  struct RealTimeType {
  };

  
  struct TimeTypeSpecs {
    // the enum
    TimeType type;
    // the name
    std::string name;
    // begin, end, and invalid 
    Time_t beginValue;
    Time_t endValue;
    Time_t invalidValue;
    
  }; 

  
  template<> struct RealTimeType<runnumber> {
    typedef unsigned int type; 
  };

  template<> struct RealTimeType<timestamp> {
    typedef unsigned long long type; 
  };

  template<> struct RealTimeType<lumiid> {
    typedef unsigned int type; 
  };

  template<> struct RealTimeType<userid> {
    typedef unsigned long long type; 
  };

  
  template<TimeType type>
  struct TimeTypeTraits {
    static  const TimeTypeSpecs & specs() {
      static const TimeTypeSpecs local = { 
	type,
	timeTypeNames[type],
	1,
	std::numeric_limits<typename RealTimeType<type>::type>::max(),
	0
      };
      return local;
    }
  };

  const TimeTypeSpecs timeTypeSpec[] = {
    TimeTypeTraits<runnumber>::specs(),
    TimeTypeTraits<timestamp>::specs(),
    TimeTypeTraits<lumiid>::specs(),
    TimeTypeTraits<userid>::specs(),
  };

}



#include<iostream>
int main() {
  using namespace cond;
  for (size_t i=0; i<TIMETYPE_LIST_MAX; i++) 
    std::cout << "Time Specs:" 
	      << " enum " << timeTypeSpec[i].type
	      << ", name " << timeTypeSpec[i].name
	      << ", begin " << timeTypeSpec[i].beginValue
	      << ", end " << timeTypeSpec[i].endValue
	      << ", invalid " << timeTypeSpec[i].invalidValue
	      << std::endl;

  return 0;
}
