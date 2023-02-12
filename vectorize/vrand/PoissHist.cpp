#include <cstdint>
#include<cmath>
#include<vector>
#include<cassert>

#include<iostream>

template<typename T>
class PoissonHist {
public:
  PoissonHist(T mu) : m_mu(mu), m_expmmu(std::exp(-mu)) {
    auto poiss = m_expmmu;
    m_cumulative.push_back(m_expmmu);
    for(;;) {
      poiss = poiss*mu/T(size());
      if (poiss <=std::numeric_limits<T>::epsilon()) break;
      auto val = m_cumulative.back() + poiss;
      if (val >= T(1.)) break;
      m_cumulative.push_back(val); 
    } 
    m_quantile[0] = findSlow(0.25);
    m_quantile[1] = findSlow(0.5);
    m_quantile[2] = findSlow(0.75);
    m_quantile[3] = findSlow(0.95); 
    m_quantile[4] = findSlow(0.99);
   }

  unsigned int findSlow(T x) const {
    auto bin = std::lower_bound(m_cumulative.begin(),m_cumulative.end(),x);
    return bin-m_cumulative.begin();
  }

  unsigned find(T x) const {
     int min  = x<T(0.5) ? 
        (x<T(0.25) ? 0 : m_quantile[0]) :
        (x<T(0.75) ? m_quantile[1] : ( x<T(0.95) ? m_quantile[2] : ( x<T(0.99) ? m_quantile[3] : m_quantile[4]))); 
     int max  = x<T(0.5) ?
        (x<T(0.25) ? m_quantile[0] : m_quantile[1]) :
        (x<T(0.75) ? m_quantile[2] : ( x<T(0.95) ? m_quantile[3] :  ( x<T(0.99) ? m_quantile[4] : size()-1)));
     max++;    
     assert(min==0 || x>=m_cumulative[min-1]);
     assert(max==size() || x<m_cumulative[max]);
     auto bin = std::lower_bound(m_cumulative.begin()+min,m_cumulative.begin()+max,x);
     return bin-m_cumulative.begin();


  }


  auto size() const { return m_cumulative.size(); }

// private:
  std::vector<T> m_cumulative;
  int m_quantile[5] = {0,0,0,0,0};
  T const m_mu;
  T const m_expmmu;
};



#include<iostream>

int main() {

for (float x = .1; x<10; x+=0.5) { 
  PoissonHist<float> f(x);
  PoissonHist<double> d(x);

  std::cout << "f " << x << ' ' << f.size() << ' ' << f.m_cumulative[0] << ' ' << f.m_cumulative.back();
  for (auto q : f.m_quantile) std::cout << ' ' << q;
  std::cout << std::endl;  
  std::cout << "d " << x << ' ' << d.size() << ' ' << f.m_cumulative[0] << ' ' << d.m_cumulative.back();
  for (auto q : d.m_quantile) std::cout << ' ' << q;
  std::cout << std::endl;

  for (float y = 0; y<1.; y+=0.1) {
    auto i1 = f.find(y);
    auto i2 = f.findSlow(y);
    assert(i1==i2);
    assert(i1==f.size() || y<f.m_cumulative[i1]);
    assert(i1==0 || y>f.m_cumulative[i1-1]);
  }

}






  return 0;
}
