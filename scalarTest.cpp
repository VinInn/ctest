//~/apps/gcc-4.8.0-bin/bin/g++ -g -march=native -O3 --std=c++11 -o scalarTest scalarTest.cpp
// when -march=native is used on an AVX machine, AVX commands are used to do even simple copy instructions


#include <iostream>
#include <vector>
#include <chrono>


double thisComputeScalar( std::vector < double > const& inp, std::vector < double > & out)
{
    double   const* pInp = & inp.front();
     double * pOut = & out.front();
    const size_t theSize = inp.size();

    // auto-vectorized with O3 & avx
    for ( size_t i = 0; i < theSize; i++ )
    {
        pOut [ i ] =    0.1 + 
                        0.2 * pInp[ i ] + 
                        0.3 * pInp[ i ] * pInp[ i ] + 
                        0.4 * pInp[ i ] * pInp[ i ] * pInp[ i ] +
                        0.5 * pInp[ i ] * pInp[ i ] * pInp[ i ] * pInp[ i ] +
                        0.6 * pInp[ i ] * pInp[ i ] * pInp[ i ] * pInp[ i ] * pInp[ i ] +
                        0.7 * pInp[ i ] * pInp[ i ] * pInp[ i ] * pInp[ i ] * pInp[ i ] * pInp[ i ] +
                        0.8 * pInp[ i ] * pInp[ i ] * pInp[ i ] * pInp[ i ] * pInp[ i ] * pInp[ i ] * pInp[ i ];
    }
    
    return 0.0f;
}


double thisComputeSuperScalar( std::vector < double > const& inp, std::vector < double > & out)
{
    double const* pInp = & inp.front();
    double * pOut = & out.front();

    // http://en.wikipedia.org/wiki/Estrin%27s_scheme 
    // auto-vectorized with O3 & avx
    for ( size_t i = 0; i < inp.size(); i++ )
    {
        pOut [ i ] =    (0.1 + 0.2 * pInp[ i ]) + 
                        (0.3 + 0.4 * pInp[ i ]) * pInp[ i ] * pInp[ i ] + 
                        ((0.5 + 0.6 * pInp[ i ]) + (0.7 + 0.8 * pInp[ i ]) * pInp[ i ] * pInp[ i ] ) 
                             * pInp[ i ] * pInp[ i ] * pInp[ i ] * pInp[ i ];
    }
    
    return 0.0f;
}


int main () 
{
    const size_t amount = 10000000;
    std::vector< double > listOne( amount );
    std::vector< double > listOut( amount );
    size_t i = 0;
    for ( auto & e : listOne )
    {
        e = i;
        i++;
    }
    auto start = std::chrono::high_resolution_clock::now();
    thisComputeScalar ( listOne, listOut );
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Test output: " <<  listOut [ 5 ] << std::endl;
    
    std::cout << "Runtime Scalar: " << std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    thisComputeSuperScalar ( listOne, listOut );
    end = std::chrono::high_resolution_clock::now();
    std::cout <<  "Test output: " << listOut [ 5 ] << std::endl;

    std::cout << "Runtime SuperScalar: " << std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() << " ms" << std::endl;


    return 0;
}
