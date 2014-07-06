#include<iostream>
#include<vector>
#include<cmath>

struct Double32 {
	Double32(){}
	template<typename T>
	Double32(T t) : d(t){}
	template<typename T>
	operator T () const { return T(d);}
	operator double() const { return d;}
    operator double&() { return d;}
	double d;
};

template<typename T> 
struct V {
	T t;
};


int main() {
{

	Double32 d, d1(-12.3);
	d = 32.4;
	int i=4;
	double p = i+d+i;
	int l = d;
	d=i;
	
	std::vector<Double32> v;
	
	v.push_back(p);
	v.push_back(i);
	
	std::cout << p << " " << double(d) << " " << i << " " << l << std::endl;
	std::cout << std::sqrt(d) << " " << std::min(double(i),p) << " " << double(std::min(d1,d)) << std::endl;
	
}	
{	
	union D {
		double d;
		float f[2];
	};
	// double d=0.000547961;
	double d=8.12623123456789e-08;
	float f = float(d);
	double e=f;
	std::cout << "ori d " << d << " float " << f << " new d " << e << std::endl;
	D du; du.d =d;
	std::cout << "D d " << du.d << " D f " << du.f[0] << " " << du.f[1]<< std::endl;
} 	

	return 0;

}