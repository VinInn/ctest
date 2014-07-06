#ifndef TrackerStablePhiSort_H
#define TrackerStablePhiSort_H
// #include "FWCore/MessageLogger/interface/MessageLogger.h"

// because of cout
#include<iostream>

#include <utility>
#include <vector>
#include <algorithm>

#include <cmath>

//#define DEBUG
namespace {
  template<class T, class Scalar>
  struct LessPair {
    typedef std::pair<T,Scalar> SortPair;
    bool operator()( const SortPair& a, const SortPair& b) {
      return a.second < b.second;
    }
  };
}


template<class RandomAccessIterator, class Extractor>
void TrackerStablePhiSort(RandomAccessIterator begin,
			    RandomAccessIterator end,
			    const Extractor& extr) {

  typedef typename Extractor::result_type        Scalar;
  typedef std::pair<RandomAccessIterator,Scalar> SortPair;

  std::vector<SortPair> tmpvec; 
  tmpvec.reserve(end-begin);

  std::vector<SortPair> tmpcop; 
  tmpcop.reserve(end-begin);

  // tmpvec holds iterators - does not copy the real objects
  for (RandomAccessIterator i=begin; i!=end; i++) {
    tmpvec.push_back(SortPair(i,extr(*i)));
  }
  
  std::sort(tmpvec.begin(), tmpvec.end(),
	    LessPair<RandomAccessIterator,Scalar>());    

  //stability check
  double pi = 3.141592653592;
  unsigned int vecSize = tmpvec.size();
  
  // create a copy of the sorted vector (work with it)
  for(unsigned int i = 0; i <vecSize; i++){
    tmpcop.push_back(tmpvec[i]);
    //   std::cout << "TrackerStablePhiSort::phi = " << tmpvec[i].second << std::endl;
  }
  
  // check if the last element is too near to zero --> probably it is zero
  double tolerance = 0.000001;
  if( fabs(tmpvec[vecSize-1].second - 0) < tolerance       // near 0
      ||
      fabs(tmpvec[vecSize-1].second - 2*pi) < tolerance ) { // near 2pi
    tmpcop.insert(tmpcop.begin(),tmpvec[vecSize-1]);
  }
  
  // special tratment of the TEC modules of rings in petals near phi=0
  // there are at most 5 modules, no other structure has less than ~10 elements to order in phi
  // hence the special case in phi~0 if the size of the elements to order is <=5
  unsigned int nMaxModulesPerRing = 5;
  bool phiZeroCase = true;
  //
  double phiMin = pi/4;
  double phiMax = 2*pi-phiMin;
  //
  // copy for [0,pi/2] elements
  std::vector<SortPair> tmppos; 
  tmppos.reserve(end-begin);
  // copy for [3pi/2,2pi] elements --> [-pi/2,0]
  std::vector<SortPair> tmpneg; 
  tmpneg.reserve(end-begin);
  //
  if( vecSize <= nMaxModulesPerRing ) {
    // check if all the elements have phi<phiMin or phi>phiMax to be sure we are near phi~0 (angles are in [0,2pi) range)
    for(unsigned int i = 0; i <vecSize; i++){
      // if a phi goes out from [0,phiMin]U[phiMax,2pi) it is not the case
      if( tmpvec[i].second > phiMin && tmpvec[i].second < phiMax ) {
	phiZeroCase = false; // if a |phi| goes out from [0,phiMin] it is not the case
      }
    }
    // go on if this is the petal phi~0 case, restricted to the case where all the |phi| are in range [0,phiMin]
    std::cout << "TrackerStablePhiSort::phiZeroCase = " << phiZeroCase << std::endl;
    if(phiZeroCase) {
      // fill the 'negative' and 'positive' vectors
      for(unsigned int i = 0; i <vecSize; i++){
	if( tmpvec[i].second > phiMin ) {
	  tmpneg.push_back(tmpvec[i]); // if a phi goes out from [phiMax,2pi) it is not the case
	} else {
	  tmppos.push_back(tmpvec[i]); // if a phi goes out from [0,phiMin] it is not the case
	}
      }
      // in this case the ordering must be: ('negative' values, >) and then ('positive' values, >) in (-pi,pi] mapping
      // sort the 'negative' values [phiMax,2pi]
      std::sort(tmpneg.begin(), tmpneg.end(),
		LessPair<RandomAccessIterator,Scalar>());    
      // sort the 'positive' values [0,phiMin]
      std::sort(tmppos.begin(), tmppos.end(),
		LessPair<RandomAccessIterator,Scalar>());    
      // fill the vector 'negative'+'positive'
      tmpcop.clear();
      for(unsigned int i = 0; i <tmpneg.size(); i++){
	tmpcop.push_back(tmpneg[i]);
	std::cout << "TrackerStablePhiSort:: neg phi = " << tmpneg[i].second << std::endl;
      }
      for(unsigned int i = 0; i <tmppos.size(); i++){
	tmpcop.push_back(tmppos[i]);
	std::cout << "TrackerStablePhiSort:: pos phi = " << tmppos[i].second << std::endl;
      }
    }
  }
  

  // overwrite the input range with the sorted values
  // copy of input container not necessary, but tricky to avoid
  std::vector<typename std::iterator_traits<RandomAccessIterator>::value_type> tmpcopy(begin,end);
  for (unsigned int i=0; i<vecSize; i++) {
    *(begin+i) = tmpcopy[tmpcop[i].first - begin];
  }
  
}

#endif

