#include<vector>
#include<algorithm>
#include<ctime>
#include<cmath>
#include "RealTime.h"

#include<iostream>
#include<boost/bind.hpp>


inline double radius(double x, double y) {
  return std::sqrt(std::pow(x,2)+std::pow(y,2));
}

class V {
public:
  V():m_x(1.),m_y(2.){}
  inline double x() const { return m_x;}
  inline double y() const { return m_y;}

  inline double r() const { return std::sqrt(m_x*m_x+m_y*m_y);}
  inline double rp() const { return std::sqrt(std::pow(m_x,2)+std::pow(m_y,2));}
  inline double rpi() const { return std::sqrt(std::pow(x(),2)+std::pow(y(),2));}

private:
  double m_x;
  double m_y;

};


int main() {

  typedef std::vector<V> VV;
  typedef std::vector<double> VD;

  VV  vv(10000000);
  VD  vd(vv.size());

  perftools::TimeType start = perftools::realTime();


  {
    double s = std::clock();
    for (int i=0; i<vv.size(); i++)
      vd[i]=vv[i].r();   
    double e = std::clock()-s;
    std::cout << "naive " << e/1000 << std::endl;
  }
  {
    double s = std::clock();
    for (int i=0; i<vv.size(); i++)
      vd[i]=vv[i].rp();   
    double e = std::clock()-s;
    std::cout << "naive pow " << e/1000 << std::endl;
  }
  {
    double s = std::clock();
    for (int i=0; i<vv.size(); i++)
      vd[i]=radius(vv[i].x(), vv[i].y());  
    double e = std::clock()-s;
    std::cout << "ext fun " << e/1000 << std::endl;
  }

  {
    double s = std::clock();
    std::transform(vv.begin(),vv.end(),vd.begin(),boost::bind(&V::r,_1));
    double e = std::clock()-s;
    std::cout << "transform method " << e/1000 << std::endl;
  }
  {
    double s = std::clock();
    std::transform(vv.begin(),vv.end(),vd.begin(),boost::bind(&V::r,_1));
    double e = std::clock()-s;
    std::cout << "transform rpi " << e/1000 << std::endl;
  }

  {
    double s = std::clock();
    std::transform(vv.begin(),vv.end(),vd.begin(),boost::bind(radius,
							      boost::bind(&V::x,_1),
							      boost::bind(&V::y,_1)));
    double e = std::clock()-s;
    std::cout << "transform ext fun " << e/1000 << std::endl;
  }

  perftools::TimeType end = perftools::realTime();

  std::cout << "tot real time " << end-start << std::endl;

  return 0;
}
