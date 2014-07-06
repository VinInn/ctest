/// Used to switch between different type of interpretations of the data (64 bits)
typedef union {
  double d;
  int i[2];
  long long ll;
  unsigned short s[4];
} ieee754;

//------------------------------------------------------------------------------

/// Converts an unsigned long long to a double
inline double ll2d(unsigned long long x) {
  ieee754 tmp;
  tmp.ll=x;
  return tmp.d;
}

//------------------------------------------------------------------------------

/// Converts a double to an unsigned long long
inline unsigned long long d2ll(double x) {
  ieee754 tmp;
  tmp.d=x;
  return tmp.ll;
}

// Taken from from quake and remixed :-)
inline double fast_isqrt_general(double x, const unsigned short ISQRT_ITERATIONS) { 

  double x2 = x * 0.5;
  double y  = x;
  unsigned long long i  = d2ll(y);
  // Evil!
  i  = 0x5fe6eb50c7aa19f9LL  - ( i >> 1 );
  y  = ll2d(i);
  for (unsigned int j=0;j<ISQRT_ITERATIONS;++j)
      y *= 1.5 - ( x2 * y * y ) ;

  return y;
}



//------------------------------------------------------------------------------

// Four iterations
inline double fast_isqrt(double x) {return fast_isqrt_general(x,4);} 

// Three iterations
inline double fast_approx_isqrt(double x) {return fast_isqrt_general(x,3);}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

#include<iostream>
#include<string>
#include<cmath>
#include<cstdlib>

// some constant
double ap2, hb0, hlova, ainv,coeff, prm[9];

void init (std::string fld="4_0T") {
  double p1[]={4.90541,17.8768,2.02355,0.0210538,0.000321885,2.37511,0.00326725,2.07656,1.71879}; // 2.0T-2G
  double p2[]={4.41982,15.7732,3.02621,0.0197814,0.000515759,2.43385,0.00584258,2.11333,1.76079}; // 3.0T-2G
  double p3[]={4.30161,15.2586,3.51926,0.0183494,0.000606773,2.45110,0.00709986,2.12161,1.77038}; // 3.5T-2G
  double p4[]={4.24326,15.0201,3.81492,0.0178712,0.000656527,2.45818,0.00778695,2.12500,1.77436}; // 3.8T-2G
  double p5[]={4.21136,14.8824,4.01683,0.0175932,0.000695541,2.45311,0.00813447,2.11688,1.76076}; // 4.0T-2G
  prm[0]=0;
  if (fld=="2_0T") for (int i=0; i<9; i++) prm[i]=p1[i];
  if (fld=="3_0T") for (int i=0; i<9; i++) prm[i]=p2[i];
  if (fld=="3_5T") for (int i=0; i<9; i++) prm[i]=p3[i];
  if (fld=="3_8T") for (int i=0; i<9; i++) prm[i]=p4[i];
  if (fld=="4_0T") for (int i=0; i<9; i++) prm[i]=p5[i];
  //  cout<<std::endl<<"Instantiation of TkBfield with key "<<fld<<endl;
  if (!prm[0]) {
    std::cout << "BadParameters" << "Undefined key - " // abort!\n";
	      <<"Defined keys are: \"2_0T\" \"3_0T\" \"3_5T\" \"3_8T\" and \"4_0T\"\n" << std::endl;
    exit(1);
  }
  ap2=4*prm[0]*prm[0]/(prm[1]*prm[1]);  
  hb0=0.5*prm[2]*std::sqrt(1.0+ap2);
  hlova=1/std::sqrt(ap2);
  ainv=2*hlova/prm[1];
  coeff=1/(prm[8]*prm[8]);
}



inline void std_ffunkti(double u, double * __restrict__ ff) {
  // Function and its 3 derivatives
  double a,b,a2,u2;
  u2=u*u; 
  a= 1/(1+u2);
  a2=-3*a*a;
  b=std::sqrt(a);
  ff[0]=u*b;
  ff[1]=a*b;
  ff[2]=a2*ff[0];
  ff[3]=a2*ff[1]*(1-4*u2);
}


inline void fast_ffunkti(double u, double * __restrict__ ff) {
  // Function and its 3 derivatives
  double a,b,a2,u2;
  u2=u*u;
#ifdef APPROX
  b = fast_approx_isqrt(1+u2);
#else
  b = fast_isqrt(1+u2);
#endif
  a= b*b;
  a2=-3*a*a;
  ff[0]=u*b;
  ff[1]=a*b;
  ff[2]=a2*ff[0];
  ff[3]=a2*ff[1]*(1-4*u2);
}


#ifdef GO_FAST
#define ffunkti fast_ffunkti
#else
#define ffunkti std_ffunkti
#endif

// cylindrical magnetic field.
inline void Bcyl(double r, double z, double * __restrict__ Bw) {
  double az=std::abs(z);
  double zainv=z*ainv;
  double u=hlova-zainv;
  double v=hlova+zainv;
  double fu[4],gv[4];
  ffunkti(u,fu);
  ffunkti(v,gv);
  double rat=0.5*r*ainv;
  double rat2=rat*rat;
  Bw[0]=hb0*rat*(fu[1]-gv[1]-(fu[3]-gv[3])*rat2*0.5);
  Bw[1]=0;
  Bw[2]=hb0*(fu[0]+gv[0]-(fu[2]+gv[2])*rat2);
}


#include <iostream>
#ifdef _WIN32
 #include <time.h>
#else
 #include <sys/time.h>
#endif

double clock_it(void)
{
#ifdef _WIN32
   clock_t start;
   double  duration;

   start = clock();
   duration = (double)(start) / CLOCKS_PER_SEC;
   return duration;
#else
	double duration = 0.0;
	struct timeval start;

	gettimeofday(&start, NULL);
	duration = (double)(start.tv_sec + start.tv_usec/1000000.0);
	return duration;
#endif
}


#define NLOOP 5000

int main() {
  double startTime, endTime, execTime;
  double Bw[3];
  double sum0=0., sum2=0.;
  double inc = 1./(double)(NLOOP);
  std::string fld="4_0T";
  init(fld);
  // warm up
  for (int i = -NLOOP/10; i <= NLOOP/10; i++) {
	  double r = (double)(i) * inc;
	  for (int j = -NLOOP/10; j <= NLOOP/10; j++) {
		  double z = (double)(j) * inc;
          Bcyl(r,z,Bw);
          sum0 += Bw[0];
          sum2 += Bw[2];
	  }
  }

  startTime = clock_it();
  for (int i = -NLOOP; i <= NLOOP; i++) {
	  double r = (double)(i) * inc;
	  for (int j = -NLOOP; j <= NLOOP; j++) {
		  double z = (double)(j) * inc;
          Bcyl(r,z,Bw);
          sum0 += Bw[0];
          sum2 += Bw[2];
	  }
  }
  endTime  = clock_it();
  execTime = endTime - startTime;
  std::cout << "start & end times: " << startTime << "  " << endTime << std::endl;
  std::cout << "time taken: " << execTime << std::endl;
  std::cout.precision(16);
  std::cout << "results: " << sum0 << "  " << sum2 << std::endl;
 }
