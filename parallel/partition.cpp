#include <iostream>
#include <atomic>
#include <cmath>
#include <omp.h>

#include <x86intrin.h>

#include <mutex>
typedef std::mutex Mutex;
// typedef std::lock_guard<std::mutex> Lock;
typedef std::unique_lock<std::mutex> Lock;

namespace global {
  // control cout....
  Mutex coutLock;
}

static unsigned long long rdtsc() {
  unsigned int taux=0;
  return __rdtscp(&taux);
}


inline 
double erfinv_like(double w) {
  w = w - 2.500000;
  float p = 2.81022636e-08;
  p = 3.43273939e-07 + p*w;
  p = -3.5233877e-06 + p*w;
  p = -4.39150654e-06 + p*w;
  p = 0.00021858087 + p*w;
  p = -0.00125372503 + p*w;
  p = -0.00417768164 + p*w;
  p = 0.246640727 + p*w;
  p = 1.50140941 + p*w;
  return p;
}

inline
double runparallel(double v, int n) {
   double s=0;
   for(int i=0; i!=n; ++i) {
     s+=erfinv_like(0.01*(v++));
     //    s+=std::sqrt(0.01*(v++));
   }
   return s;
}



void goStatic() {
 
  //    double * data = 0; // new double[800000];

  int tot = 1000000;

  struct alignas(128) AlDouble { double x;};
  AlDouble resv[omp_get_max_threads()]{0,};


  long long t = 0;

   for (int i=0; i!=1000; ++i) {
     t -= rdtsc();
#pragma omp parallel
    {
       int n = omp_get_num_threads();
       int k = omp_get_thread_num();
       auto ln = tot/n;
       double v = k*ln;
       resv[k].x += runparallel(v,ln);       
    }
    t += rdtsc();

   } 
   double res=0;
   for ( auto v : resv) res+=v.x;
   std::cout << t << " " << res << std::endl;
   
   //  delete [] data;
   
}


void goDynamic() {
 
  //    double * data = 0; // new double[800000];


  int tot =   1000000;
  int block = 10000;
  //struct alignas(128) AlDouble { double x;};
  struct alignas(double) AlDouble { double x;};

  AlDouble resv[omp_get_max_threads()]{0,};

  long long t = 0;


  for (int i=0; i!=1000; ++i) {
    t -= rdtsc();
    std::atomic<int> start(0);
  
#pragma omp parallel
    {
      int k = omp_get_thread_num();
      while (true) {
	int ls = start; 
	if (ls>=tot) break;
	/*
	{
	  Lock l(global::coutLock);
	  std::cout  <<"before "<< k << ": "<< ls << std::endl;
	}
	*/
	while (ls<tot && !std::atomic_compare_exchange_weak(&start,&ls,ls+block)); 
	// int n = omp_get_num_threads();
	auto ln = std::min(block,tot-ls);
	if (ln<=0) break;
	double v = ls;
	resv[k].x += runparallel(v,ln);
	/*
	{
	  Lock l(global::coutLock);
	  std::cout <<"after "<< k << ": " << ls << " " << ln<< std::endl;
	}
	*/
      }
    }
    t += rdtsc();

  } 
   double res=0;
   for ( auto v : resv) res+=v.x;
   std::cout << t << " " << res << std::endl;

   
   //  delete [] data;
   
}


int main() {

  goStatic();
  goDynamic();

  goStatic();
  goDynamic();

  return 0;
}
