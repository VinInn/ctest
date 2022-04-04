#include <iostream>
#include <cmath>
#include <cfloat>
#include<vector>

int main(int argc, char**)
{


    std::vector<double> v = {NAN,INFINITY,0.0,std::exp(800),DBL_MIN/2.0};

    std::cout << "compile time\n" << std::boolalpha
              << "isfinite(NaN) = " << std::isfinite(NAN) << '\n'
              << "isfinite(Inf) = " << std::isfinite(INFINITY) << '\n'
              << "isfinite(0.0) = " << std::isfinite(0.0) << '\n'
              << "isfinite(exp(800)) = " << std::isfinite(std::exp(800)) << '\n'
              << "isfinite(DBL_MIN/2.0) = " << std::isfinite(DBL_MIN/2.0) << '\n';


    std::cout << "run time\n" << std::boolalpha
              << "isfinite(NaN) = " << std::isfinite(v[0]) << '\n'
              << "isfinite(Inf) = " << std::isfinite(v[1]) << '\n'
              << "isfinite(0.0) = " << std::isfinite(v[2]) << '\n'
              << "isfinite(exp(800)) = " << std::isfinite(v[3]) << '\n'
              << "isfinite(DBL_MIN/2.0) = " << std::isfinite(v[4]) << '\n';

   float x =  1.0f - float(argc);

    std::cout << std::boolalpha
              << "x = " << x << '\n'
              << "isfinite(0.f/x) = " << std::isfinite(0.f/x) << '\n'
              << "isfinite(1.f/x) = " << std::isfinite(1.f/x) << '\n'
              << "isfinite(x) = " << std::isfinite(0.0) << '\n'
              << "isfinite(exp(x)) = " << std::isfinite(std::exp(x)) << '\n'
              << "isfinite(DBL_MIN/x) = " << std::isfinite(DBL_MIN/2-x) << '\n';


}
