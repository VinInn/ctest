#include <iostream>
#include <vector>
#include <algorithm>
#include <boost/thread/thread.hpp>
#include <cmath>
#include <limits>
#include<queue>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
typedef boost::mutex::scoped_lock ScopedLock;


template<typename I, typename T> 
class ParallelFor {
public:
  typedef ParallelFor<I,T> self;
  typedef I Iterator;
  typedef std::pair<Iterator,Iterator> Range;
  typedef T Value;
  typedef size_t Distance;
  typedef boost::function<void(T&)> Function;

  template<typename F>
  Parallel(Iterator b, Iterator e, F f, size_t nThreads, Distance stride) :
    m_nThreads(nThreads), m_stride(stride),
    m_end(e), m_function(f), 
    m_current(b)
  {

  }

  void compute() {
    boost::thread_group threads;
    for (size_t i=0; i<m_nThreads; ++i)
      threads.create_thread(boost::bind(&self::iterate,boost::ref(*this)));
    threads.join_all();
  }

private:

  Range next() {
    Range ret;
    ScopedLock gl(lock);
    // get next task
    ret.first=m_current;
    m_current+=m_stride;
    ret.second = std::min(m_current,m_end);
    return ret;
  }

  void iterate() {
    while (true) {
      Range range = next();
      if (!(range.first<range.second)) return;
      for (Iterator p=range.first; p!=range.second; ++p)
	m_function(*p);
    }
  }


private:
  boost::mutex lock;

  size_t m_nThreads;
  Distance m_stride;

  Iterator m_end;
  Function m_function;
  
  Iterator m_current;



};

template<typename T> 
class ParallelNext {
public:
  typedef ParallelNext<T> self;
  typedef T Value;
  typedef boost::function<void(T&)> Function;

  template<typename F>
  Parallel(T & value, F f, size_t nThreads) :
    m_nThreads(nThreads)
    m_function(f), 
    m_current(value)
    m_state(true)
  {

  }

  void compute() {
    boost::thread_group threads;
    for (size_t i=0; i<m_nThreads; ++i)
      threads.create_thread(boost::bind(&self::iterate,boost::ref(*this)));
    threads.join_all();
  }

private:

  void iterate() {
    while (true) {
      ScopedLock gl(lock);
      if (!m_state) return;
      Value v=m_current;
      m_state = m_current.next();
      gl.unlock();
      m_function(v);
    }
  }


private:
  boost::mutex lock;

  size_t m_nThreads;
  Function m_function;
  
  Value & m_current;
  bool m_state;


};


namespace {
  
  /* default range of numbers to search */
  const int rangeStart = 1;
  const int rangeEnd  = 10000000;
  
  const int nThread = 4;
  const int stride = 1000;
  
  typedef std::vector<std::pair<int,bool> > Buffer; 
  
  
  struct Compute {
    void operator()(std::pair<int,bool> & p) {
      int index = p.first;
      p.second=true; /* assume number is prime */
      int limit = (int) sqrt((float)index) + 1;
      int factor = 3;
      while (p.second && (factor <= limit)) {
	if (index%factor == 0) p.second = false;
	factor += 2;
      }
    }
  };

  struct Insert {
    std::vector<int> & allPrimes;
    Insert( std::vector<int> & v) : allPrimes(v){}
    void operator()(Buffer::value_type p) {
      if (p.second) allPrimes.push_back(p.first);
    }

  };

}




int main() {

  Buffer allNumbers;
  allNumbers.reserve((rangeEnd-rangeStart)/2);
  int start = std::max(3,rangeStart);
  if (start%2==0) start++;
  for (int i=start; i<rangeEnd;i+=2)
    allNumbers.push_back(std::make_pair(i,true));

  ParallelFor<Buffer::iterator,Buffer::value_type> 
    p(allNumbers.begin(),allNumbers.end(),Compute(),4,1000);
  
  p.compute();
  std::vector<int> allPrimes;
  allPrimes.reserve(allNumbers.size());
  std::for_each(allNumbers.begin(),allNumbers.end(),Insert(allPrimes));

  // std::sort(allPrimes.begin(),allPrimes.end());

  int number_of_primes = allPrimes.size();
  printf("\nProgram Done: %d primes found\n",   number_of_primes);
  // fprintf(stderr, "Time is %f\n",1.e-6*wt);
  const int MAX_PRINT=10;

  int n_print = number_of_primes>MAX_PRINT ? MAX_PRINT : number_of_primes;

    printf("\nFirst %d are\n",n_print);
    for( int i=0; i<n_print; i++ )
     {
       printf("%d\n",allPrimes[i]);
     }
    n_print = number_of_primes-n_print;

    n_print = n_print>MAX_PRINT ? MAX_PRINT : n_print;
    printf("\nLast %d are\n",n_print);
    for( int i=0; i<n_print; i++ )
     {
     printf("%d\n",allPrimes[number_of_primes-n_print+i]);
     }


  return 0;
}
