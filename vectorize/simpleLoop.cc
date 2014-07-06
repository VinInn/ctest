#include<cstddef>

void loop1( double const * __restrict__ x_in,  double * __restrict__ x_out, double const * __restrict__ c, int N) { 
   for(int i=0; i!=N; ++i)
       x_out[i] = c[i]*x_in[i];
}



void loop2( double const * __restrict__ x_in,  double * __restrict__ x_out, double const * __restrict__ c, unsigned int N) {
   for(unsigned int i=0; i!=N; ++i)
       x_out[i] = c[i]*x_in[i];
}

void loop21( double const * __restrict__ x_in,  double * __restrict__ x_out, double const * __restrict__ c, size_t N) {
   for(size_t i=0; i!=N; ++i)
       x_out[i] = c[i]*x_in[i];
}

void loop21( double const * __restrict__ x_in,  double * __restrict__ x_out, double const * __restrict__ c, unsigned long long N) {
   for(unsigned long long  i=0; i!=N; ++i)
       x_out[i] = c[i]*x_in[i];
}


void loop3( double const * __restrict__ x_in,  double * __restrict__ x_out, double const * __restrict__ c, size_t N) {
   double const * end = x_in+N;
   for(; x_in!=end; ++x_in, ++x_out, ++c)
       (*x_out) = (*c) * (*x_in);
}

void loop10( double const * __restrict__ x_in,  double * __restrict__ x_out, double const * __restrict__ c) {
   for(unsigned int i=0; i!=10; ++i)
       x_out[i] = c[i]*x_in[i];
}


template<typename T, unsigned int N>
void loopTu( T const * __restrict__ x_in,  T * __restrict__ x_out, T const * __restrict__ c) {
   for(unsigned int i=0; i!=N; ++i)
       x_out[i] = c[i]*x_in[i];
}

template<typename T, unsigned long long N>
void loopTull( T const * __restrict__ x_in,  T * __restrict__ x_out, T const * __restrict__ c) {
   for(unsigned long long i=0; i!=N; ++i)
       x_out[i] = c[i]*x_in[i];
}

template<typename T, unsigned int N>
T sumTu( T const * __restrict__ x_in,  T const * __restrict__ x_out, T const * __restrict__ c) {
   T r = 0;
   for(unsigned int i=0; i!=N; ++i)
       r += x_out[i] + c[i]*x_in[i];
   return r;
}

template<typename T, unsigned long long N>
T sumTull( T const * __restrict__ x_in,  T const * __restrict__ x_out, T const * __restrict__ c) {
   T r = 0;
   for(unsigned long long i=0; i!=N; ++i)
       r += x_out[i] + c[i]*x_in[i];
   return r;
}


template<typename T, unsigned int N>
T sumNu( T const * __restrict__ x_in,  T const * __restrict__ x_out, T const * __restrict__ c) {
   return x_out[N] + c[N]*x_in[N] + sumNu<T,N-1>(x_in,x_out,c);
}

template<>
double sumNu<double,0>( double const *,  double const *, double const *) {
  return 0;
}


double __attribute__ ((optimize("fast-math")))
go(double const * __restrict__ x_in,  double * __restrict__ x_out, double const * __restrict__ c) {
 
  loopTu<double,10>(x_in, x_out, c);
  double a = sumTu<double,10>(x_in, x_out, c);

  loopTull<double,10>(x_in, x_out, c);
  a += sumTull<double,10>(x_in, x_out, c);

  a += sumNu<double,10>(x_in, x_out, c);


  return a;
}


