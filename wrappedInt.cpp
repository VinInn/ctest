#include<limits>
#include<cmath>
#include<iomanip>
#include<iostream>


template<typename T, int Numerator, int Denominator>
class WrappedInt {
public:
  using value_type=T;
  usign self=WrappedInt<T, Numerator,Denominator>;
  
  static constepxr double dmax = double(Numerator)/double(Denominator);
  static constepxr float abin = std::numeric_limits<value_type>::max()/dmax;
  static constepxr float scale = dmax/std::numeric_limits<value_type>::max();
  static constexpr value_type f2i(float x) { return std::round(x*abin);}
  static constexpr float i2f(T x) { return x*scale;}

  WrappedInt(float x) : m_value(ftoi(x)){}

  float tof() const { return i2f(m_value);}

  value_type value() const { return m_value;}

  self & operator-=(WrappedInt<T,N,D> x) {
    m_value-=x;
    return self;
  }

  self & operator+=(WrappedInt<T,N,D> x) {
    m_value+=x;
    return self;
  }


  
private:
  value_type m_value;
};

template<typename T, int N, int D>
WrappedInt<T,N,D> 
operator-(WrappedInt<T,N,D> a, WrappedInt<T,N,D> b) {
  return a-=b;
}
	  
template<typename T, int N, int D>
WrappedInt<T,N,D> 
operator+(WrappedInt<T,N,D> a, WrappedInt<T,N,D> b) {
  return a+=b;
}

template<typename T, int N, int D>
bool
operator<(WrappedInt<T,N,D> a, WrappedInt<T,N,D> b) {
  return a.value()<b.value();
}

template<typename T, int N, int D>
bool
operator==(WrappedInt<T,N,D> a, WrappedInt<T,N,D> b) {
  return a.value()==b.value();
}




int main() {

  int imax = std::numeric_limits<int>::max();
  unsigned int umax = std::numeric_limits<unsigned int>::max();

  
  int nine = 0.9f*imax;
  int mnine = -nine;

  std::cout << std::hex << nine << ' ' << mnine << std::endl;
  
  
  unsigned int unine =  (0.9f)*umax/2.f;
  unsigned int umnine = (1.1f)*umax/2.f;

  std::cout << std::hex << unine << ' ' << umnine << std::endl;


  
  std::cout << float(nine-mnine)/imax << std::endl;
  std::cout << float(mnine-nine)/imax << std::endl;
  
  std::cout << 2.*float((unine-umnine)^0Xffffffff)/umax << std::endl;
  std::cout << 2.*float((umnine-unine)^0)/umax << std::endl;


  
  return 0;
}
