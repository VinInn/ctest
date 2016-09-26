#include <chrono>
#include <array>
#include <iostream>
#include "benchmark.h"

inline
size_t
fib(size_t n)
{
  if (n == 0)
    return n;
  if (n == 1)
    return 1;
  return fib(n - 1) + fib(n - 2);
}

inline
double perform_computation(int w) {
  return fib(w);
}

int main()
{
    using namespace std;
    auto start = chrono::high_resolution_clock::now();
    int value = 42;
    benchmark::touch(value);
    double answer = perform_computation(value);
    benchmark::keep(answer);
    auto delta = chrono::high_resolution_clock::now() - start;
    std::cout << "Answer: " << answer << ". Computation took "
              << chrono::duration_cast<chrono::milliseconds>(delta).count()
              << " ms" << std::endl;


    constexpr int N=1025;
    std::array<float,N> x,y,z;


    benchmark::touch(x);
    benchmark::touch(y);
    benchmark::touch(z);
    for (int i=0; i<N; ++i)
       z[i]=y[i]=x[i]=i*1.e-6;
    benchmark::keep(z);
  


    delta = start - start;
    for (int j=0; j<1000000; ++j) {
    delta -= (chrono::high_resolution_clock::now()-start);
    benchmark::touch(x);
    benchmark::touch(y);
    benchmark::touch(z);
    for (int i=0; i<N-1; ++i)
       z[i]+=y[i]*x[i];
    benchmark::keep(z);
    delta += (chrono::high_resolution_clock::now()-start);
    }
    std::cout << z[N-1] << " Computation took "
              << chrono::duration_cast<chrono::milliseconds>(delta).count()
              << " ms" << std::endl;

   for (int i=0; i<N-1; ++i)
       z[i]=y[i]=x[i]=i*1.e-6;


    delta = start - start;
    for (int j=0; j<1000000; ++j) {
   for (int i=0; i<N; ++i)
       z[i]=y[i]=x[i]=i*1.e-6;                                                                          
    delta -= (chrono::high_resolution_clock::now()-start);
    benchmark::touch(x);
    benchmark::touch(y);
    benchmark::touch(z);
    for (int i=0; i<N-1; ++i)
       z[i+1]+=y[i]*z[i];
    benchmark::keep(z);
    delta += (chrono::high_resolution_clock::now()-start);
    }
    std::cout << z[N-1]<<" Computation took "
              << chrono::duration_cast<chrono::milliseconds>(delta).count()
              << " ms" << std::endl;

   for (int i=0; i<N; ++i)
       z[i]=y[i]=x[i]=i*1.e-6;

   delta = start - start;
    for (int j=0; j<1000000; ++j) {
   for (int i=0; i<N-1; ++i)
       z[i]=y[i]=x[i]=i*1.e-6;                                                                          
    delta -= (chrono::high_resolution_clock::now()-start);
    benchmark::touch(x);
    benchmark::touch(y);
    benchmark::touch(z);
    for (int i=0; i<N-1; ++i)
       z[i]+=y[i]*z[i];
    benchmark::keep(z);
    delta += (chrono::high_resolution_clock::now()-start);
    }
    std::cout<< z[N-1] <<" Computation took "
              << chrono::duration_cast<chrono::milliseconds>(delta).count()
              << " ms" << std::endl;




    return 0;
}

