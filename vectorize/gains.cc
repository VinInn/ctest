#include<vector>
using uint16_t = unsigned short;
std::vector<uint16_t>  ADCs;
uint16_t firstStrip;
float * first;
inline float   gain (int strip)  {int apv = strip/128; return *(first+apv);}
void applyGains() {
  int strip = firstStrip;
  for( auto &  adc :  ADCs) {
    // if(adc > 253) continue; //saturated, do not scale
    auto charge = int( float(adc)/gain(strip++) + 0.5f ); //adding 0.5 turns truncation into rounding
//    if(adc < 254) adc = ( charge > 1022 ? 255 : 
//			  ( charge >  253 ? 254 : charge ));
    adc=charge;
  }
}
