#include <iostream>

#include <vector>
#include <array>

constexpr unsigned int iSize = 32;

template < class TType, unsigned int IPos >
class AddVars
{
public:
	inline static void add( TType const * __restrict__ var1,TType const * __restrict__ var2, TType *  __restrict__ res )
	{
		for ( unsigned int i = 0 ; i < iSize; i++ )
		{
		 	res[i] = var1[ i] * var2[i];
		}
	}
};

void __attribute__((no_inline, no_clone))
add( double const * __restrict__ var1, double const * __restrict__ var2, double *  __restrict__ res )
	{
		for ( unsigned int i = 0 ; i < iSize; i++ )
		{
		 	res[i] = var1[ i] * var2[i];
		}
	}


int main ()
{
  std::vector<double> v1(iSize, 23.0);
  std::vector<double> v2(iSize, 5.0);
  std::vector<double> res(iSize, 0.0);
  // std::array<double,iSize> v1{23.0};
  // std::array<double, iSize> v2{5.0};
  // std::array<double, iSize> res{0.0};

	double const * __restrict__ var1_ptr  = &v1.front();
	double const * __restrict__ var2_ptr  = &v2.front();
	double * __restrict__  res_ptr   = &res.front();

	// this call will be with runtime-aliasing
	AddVars<double, iSize >::add( var1_ptr, var2_ptr, res_ptr );

	for ( unsigned int i = 0; i < iSize; ++ i)
	{
		std::cout << v1[i] << " * " << v1[i] << " = " << res[i] << std::endl;
	}

	::add( var1_ptr, var2_ptr, res_ptr );

	for ( unsigned int i = 0; i < iSize; ++ i)
	{
		std::cout << v1[i] << " * " << v1[i] << " = " << res[i] << std::endl;
	}



	// this call will be runtime-aliased
	for ( unsigned int i = 0; i < iSize; ++ i)
	{
		res_ptr[i] = var1_ptr[i] * var2_ptr[i];
	}

	for ( unsigned int i = 0; i < iSize; ++ i)
	{
		std::cout << v1[i] << " * " << v1[i] << " = " << res[i] << std::endl;
	}


	return 0;
}
