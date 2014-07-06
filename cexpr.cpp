#include<cmath>
#include<algorithm>


#include<iostream>

float ntmp_ori[256];
inline void computeShape() {
  // pulse shape time constants in ns
  const float ts1  = 8.;          // scintillation time constants : 1,2,3
  const float ts2  = 10.;           
  const float ts3  = 29.3;         
  const float thpd = 4.;          // HPD current collection drift time
  const float tpre = 9.;          // preamp time constant (refit on TB04 data)
  
  const float wd1 = 2.;           // relative weights of decay exponents 
  const float wd2 = 0.7;
  const float wd3 = 1.;
  
  // pulse shape componnts over a range of time 0 ns to 255 ns in 1 ns steps
  int nbin = 256;
  // sh.setNBin(nbin);
  std::vector<float> ntmp(nbin,0.0);  // zeroing output pulse shape
  std::vector<float> nth(nbin,0.0);   // zeroing HPD drift shape
  std::vector<float> ntp(nbin,0.0);   // zeroing Binkley preamp shape
  std::vector<float> ntd(nbin,0.0);   // zeroing Scintillator decay shape
  
  int i,j,k;
  float norm;
  
  // HPD starts at I and rises to 2I in thpd of time
  norm=0.0;
  for(j=0;j<thpd && j<nbin;j++){
    nth[j] = 1.0 + ((float)j)/thpd;
    norm += nth[j];
  }
  // normalize integrated current to 1.0
  for(j=0;j<thpd && j<nbin;j++){
    nth[j] /= norm;
  }
  
  // Binkley shape over 6 time constants
  norm=0.0;
  for(j=0;j<6*tpre && j<nbin;j++){
    ntp[j] = ((float)j)*exp(-((float)(j*j))/(tpre*tpre));
    norm += ntp[j];
  }
  // normalize pulse area to 1.0
  for(j=0;j<6*tpre && j<nbin;j++){
    ntp[j] /= norm;
  }
  
  // ignore stochastic variation of photoelectron emission
  // <...>
  
  // effective tile plus wave-length shifter decay time over 4 time constants
  int tmax = 6 * (int)ts3;
  
  norm=0.0;
  for(j=0;j<tmax && j<nbin;j++){
    ntd[j] = wd1 * exp(-((float)j)/ts1) + 
      wd2 * exp(-((float)j)/ts2) + 
      wd3 * exp(-((float)j)/ts3) ; 
    norm += ntd[j];
  }
  // normalize pulse area to 1.0
  for(j=0;j<tmax && j<nbin;j++){
    ntd[j] /= norm;
  }
  
  int t1,t2,t3,t4;
  for(i=0;i<tmax && i<nbin;i++){
    t1 = i;
    //    t2 = t1 + top*rand;
    // ignoring jitter from optical path length
    t2 = t1;
    for(j=0;j<thpd && j<nbin;j++){
      t3 = t2 + j;
      for(k=0;k<4*tpre && k<nbin;k++){       // here "4" is set deliberately,
	t4 = t3 + k;                         // as in test fortran toy MC ...
	if(t4<nbin){                         
	  int ntb=t4;                        
	  ntmp[ntb] += ntd[i]*nth[j]*ntp[k];
	}
      }
    }
  }
  
  // normalize for 1 GeV pulse height
  norm = 0.;
  for(i=0;i<nbin;i++){
    norm += ntmp[i];
  }
  
  //cout << " Convoluted SHAPE ==============  " << endl;
  for(i=0; i<nbin; i++) {
    ntmp[i] /= norm;
    std::cout << " shape " << i << " = " << ntmp[i] << std::endl;   
    ntmp_ori[i]  = ntmp[i];
  }
  
  //  for(i=0; i<nbin; i++){
  //  sh.setShapeBin(i,ntmp[i]);
  // }
}



namespace hcalPulseShapeDetails {
  // using std::min;
  constexpr int min(int a, int b) { return a<b ? a : b;}

  // pulse shape time constants in ns
  constexpr float ts1  = 8.f;          // scintillation time constants : 1,2,3
  constexpr float ts2  = 10.f;           
  constexpr float ts3  = 29.3f;         
  constexpr float thpd = 4.f;          // HPD current collection drift time
  constexpr float tpre = 9.f;          // preamp time constant (refit on TB04 data)
  constexpr int ithpd = 4;          // HPD current collection drift time
  constexpr int itpre = 9;          // preamp time constant (refit on TB04 data)
  
  constexpr float wd1 = 2.f;           // relative weights of decay exponents 
  constexpr float wd2 = 0.7f;
  constexpr float wd3 = 1.f;
  // pulse shape componnts over a range of time 0 ns to 255 ns in 1 ns steps
  constexpr int nbin = 256;

  // HPD starts at I and rises to 2I in thpd of time
  constexpr float nth_j(int j) { return 1.0f + ((float)j)/thpd;}
  constexpr float sum_nth(int i) { return i==0 ? 0 : nth_j(i-1)+sum_nth(i-1);}
  constexpr float nth(int j) { return j<min(ithpd,nbin) ? nth_j(j)/sum_nth(min(ithpd,nbin)) : 0;}
  // Binkley shape over 6 time constants
  constexpr float ntp_j(int j) { return  ((float)j)*exp(-((float)(j*j))/(tpre*tpre));}
  constexpr float sum_ntp(int i) { return i==0 ? 0 : ntp_j(i-1)+sum_ntp(i-1);}
  constexpr float ntp(int j) { return  j<min(6*itpre,nbin) ? ntp_j(j)/sum_ntp(min(6*itpre,nbin)) : 0;}
  // effective tile plus wave-length shifter decay time over 4 time constants
  constexpr int tmax = 6 * (int)ts3;
  constexpr float ntd_j(int j) { return wd1 * exp(-((float)j)/ts1) + 
      wd2 * exp(-((float)j)/ts2) + 
      wd3 * exp(-((float)j)/ts3) ; 
  }
  constexpr float sum_ntd(int i) { return i==0 ? 0 : ntd_j(i-1)+sum_ntd(i-1);}
  constexpr float ntd(int j) { return j<min(tmax,nbin) ? ntd_j(j)/sum_ntd(min(tmax,nbin)) : 0;}
      
  
  
  /*
    constexpr float ntmp(int ntb) {
    float ret=0.f;
    for(int k=0;k<4*itpre && k<nbin;k++){       // here "4" is set deliberately  as in test fortran toy MC ...
      int t3 = ntb-k;
      for(int j=0;j<ithpd && j<nbin;j++){
	int i = t3-j;
	if (i>=0) ret += ntd(i)*nth(j)*ntp(k);
      }
    }
    return ret
  }
  */
  
  constexpr float ntmp_ijk(int i, int j, int k) { return i<0 ? 0 :  ntd(i)*nth(j)*ntp(k);}
  constexpr float ntmp_jk(int j, int k, int t3) { return j==0 ? 0 : ntmp_ijk(t3-(j-1),j-1,k)+ ntmp_jk(j-1,k,t3);}
  constexpr float ntmp_k(int k, int ntb) { return k==0 ? 0 : ntmp_jk(min(ithpd,nbin), k-1, ntb-(k-1))+ ntmp_k(k-1,ntb);}
  constexpr float ntmp_n( int ntb) { return  ntmp_k(min(4*itpre,nbin),ntb);}
  
  
  
  constexpr float sum_ntmp(int i) { return i==0 ? 0 : ntmp_n(i-1)+sum_ntmp(i-1);}
  constexpr float ntmp(int i) { return ntmp_n(i)/sum_ntmp(nbin);}


  template<int i> struct NTMP { static constexpr float value = ntmp(i);};
  
}

///////////
// Some meta template stuff
template<int...> struct indices{};

template<int I, class IndexTuple, int N>
struct make_indices_impl;

template<int I, int... Indices, int N>
struct make_indices_impl<I, indices<Indices...>, N>
{
typedef typename make_indices_impl<I + 1, indices<Indices..., I>,
N>::type type;
};

template<int N, int... Indices>
struct make_indices_impl<N, indices<Indices...>, N> {
typedef indices<Indices...> type;
};

template<int N>
struct make_indices : make_indices_impl<0, indices<>, N> {};
// end of stuff

#include <cstddef>
#include <iostream>
#include <utility>
#include <type_traits>

template<class T, std::size_t N>
struct carray {
 T data[N];
 constexpr const T& operator[](std::size_t i) { return data[i]; }
 T& operator[](std::size_t i) { return data[i]; }
 constexpr std::size_t size() { return N; }
 T* begin() { return data; }
 T* end() { return data + N; }
 const T* begin() const { return data; }
 const T* end() const { return data + N; }
};

template<int I0, class F, int... I>
constexpr carray<decltype(std::declval<F>()(std::declval<int>())), sizeof...(I)>
do_make(F f, indices<I...>)
{
return carray<decltype(std::declval<F>()(std::declval<int>())),
sizeof...(I)>{{ f(I0 + I)... }};
}

template<int N, int I0 = 0, class F>
constexpr carray<decltype(std::declval<F>()(std::declval<int>())), N>
make(F f) {
 return do_make<I0>(f, typename make_indices<N>::type());
}


/////////////




#include <iostream>

template<int i> struct printNTMP {
  static void go(){
    using namespace hcalPulseShapeDetails;
    printNTMP<i-1>::go();
    std::cout << i << ": " << NTMP<i>::value << std::endl;
  }
};
template<> struct printNTMP<0> {
  static void go(){
    using namespace hcalPulseShapeDetails;
    std::cout << 0 << ": " << NTMP<0>::value << std::endl;
  }
};

template<int i> struct initNTMP : public initNTMP<i-1>{
  constexpr initNTMP() : value(hcalPulseShapeDetails::ntmp(i)){}
  float value;
};

template<> struct initNTMP<0> {
   constexpr initNTMP() : value(hcalPulseShapeDetails::ntmp(0)){}
   float value;
};

union NTMP_V {
  constexpr NTMP_V() : values() {}
  initNTMP<255> values;
  float a[256];
};

#include <iostream>
int main() {
  
  computeShape();

  using namespace hcalPulseShapeDetails;
  /*
   sh.setNBin(nbin);
  
  for(i=0; i<nbin; i++){
    sh.setShapeBin((i,ntmp(i));
  }
  */

  printNTMP<255>::go();

  constexpr NTMP_V ntmp_a;

  std::cout << sizeof(initNTMP<255> ) << std::endl;
  std::cout << sizeof(NTMP_V) << std::endl;

  for(int i=0; i<nbin; i++)
    std::cout << i << ": " <<  ntmp_ori[i] << " " << ntmp_a.a[i] << std::endl;
  
  constexpr auto v = make<256>(ntmp); // OK
  constexpr auto e1 = v[1]; // OK
  for (auto i : v) std::cout << i << std::endl;


  return 0;

}
