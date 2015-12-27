#include<atomic>
#include<limits>
#include<iostream>

template<typename T>
inline
void update_maximum(std::atomic<T>& max_value, T const& value) noexcept
{
    T prev_value = max_value;
    while(prev_value < value &&
            !max_value.compare_exchange_weak(prev_value, value))
        ;
}

template<typename T>
inline
void update_minimum(std::atomic<T>& min_value, T const& value) noexcept
{
    T prev_value = min_value;
    while(prev_value > value &&
            !min_value.compare_exchange_weak(prev_value, value))
        ;
}


namespace debugTools {
  template<typename T, int NBin=10>
  struct Stat {
    using Value = std::atomic<T>;
    using Counter = std::atomic<long long>;
    
    DoStat(const char * iname, T imin, T imax) :
      m_name(iname), m_hmin(imin), m_hInvWith(T(NBin)/(imax-imin)),
      m_min(std::numeric_limits<T>::max()),
      m_max(std::numeric_limits<T>::min()) {
      for (auto & x : m_hist) x=0;
    }
    ~DoStat() {
      std::cout << m_name << ": " << m_min<<','<<m_max
		<< " ["<<m_hmin<<':'<<std::round(m_hmim+NBin/m_hInvWith)<<"] ";
      for (auto const & x : m_hist) std::cout << x <<',';
      std::cout << std::endl;
    }

    void operator()(T x) {
      update_minimum(m_min,x);
      update_maximum(m_max,x);
      int bin = (x-m_hmin)*m_hInvWith;
      bin = std::max(0,std::min(NBin-1,bin));
      m_hist[bin]++;
    }
    
    const char * m_name;
    const T m_hMin;
    const T m_hInvWith;
    Counter m_hist[NBin];
    Value m_min;
    Value m_max;
  };

  template<typename T, in NBin=10>
  struct FakeStat {
    void operator()(T) {}
  };

  template<typename T, int N> using Stat = DoStat<T,N>;
  
}



  

