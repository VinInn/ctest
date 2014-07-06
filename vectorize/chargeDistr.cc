#include<cmath>
#include<algorithm>
#include<vector>
#include<cassert>

void comp(float&, float &);

inline float approx_erf(float x) {
  return std::sqrt(x*x+1); // not erf of course!
}

float  Nsigma; // is 3 but....



int Nstrips; // a sort of constant///

std::vector<float> localAmplitudes;
std::vector<float> coupling;

void chargeDistribution(int N) {

  float chargePosition[N];
  float chargeSpread[N];
  int fromStrip[N];
  int nStrip[N];
  // load not vectorize
  for (int i=0; i!=N;++i) comp(chargePosition[i],chargeSpread[i]);
  // this vectorize
  for (int i=0; i!=N;++i) {
    fromStrip[i]  = std::max( 0,  int(std::floor( chargePosition[i] - Nsigma*chargeSpread[i])) );
    nStrip[i] = std::min( Nstrips, int(std::ceil( chargePosition[i] + Nsigma*chargeSpread[i])) ) - fromStrip[i];
  }
  int tot=0;
  for (int i=0; i!=N;++i) tot += nStrip[i];
  tot+=N; // add last strip 
  float value[tot];

  // assign relative position (lower bound of strip) in value;
  int kk=0;
  for (int i=0; i!=N;++i) {
    auto delta = 1.f/(std::sqrt(2.f)*chargeSpread[i]);
    auto pos = delta*(float(fromStrip[i])-chargePosition[i]);
    for (int j=0;j<=nStrip[i]; ++j)  /// include last strip
      value[kk++] = pos+float(j)*delta;  
  }
  assert(kk==tot);

  // main loop fully vectorized
  for (int k=0;k!=tot; ++k)
    value[k] = approx_erf(value[k]);

  // saturate 0 & NStrips strip to 0 and 1???
  /*  */

  // compute integral over strip (lower bound becomes the value)
  for (int k=0;k!=tot-1; ++k)
    value[k]-=value[k+1];


  float charge[Nstrips];
  kk=0;
  for (int i=0; i!=N;++i){ 
    for (int j=0;j!=nStrip[i]; ++j)
      charge[fromStrip[i]+j]+= value[kk++];
    kk++; // skip last "strip"
  }
  assert(kk==tot);


  /// do crosstalk... (can be done better, most probably not worth)
  int sc = coupling.size();
  for (int i=0;i!=Nstrips; ++i) {
    int strip = i;
    auto affectedFromStrip  = std::max( 0, strip - sc + 1);
    auto affectedUntilStrip = std::min(Nstrips, strip + sc);  
    for (auto affectedStrip=affectedFromStrip;  affectedStrip < affectedUntilStrip;  ++affectedStrip)
      localAmplitudes[affectedStrip] += charge[i] * coupling[std::abs(affectedStrip - strip)] ;
  }

}
