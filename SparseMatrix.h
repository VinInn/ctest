#ifndef MathSparseMatrix_H
#define MathSparseMatrix_H

#include "Math/SVector.h"
#include "Math/SMatrix.h"

#include<algorithm>
#include<cassert>;

namespace  sparseMatrixDetails {
  struct Index {
    Index(){}
    Index(int ii, int jj) : i(ii),j(jj){}
    int i; int j;
    bool operator==(Index rh) const { return i==rh.i && j==rh.j;}
    bool operator<(Index rh) const { return i<rh.i || ( !(rh.i<i) && j<rh.j);}
    void transpose() { std::swap(i,j);}
  };

  template<typename T>
  struct Value {
    Value(){}
    Value(int ii, int jj, T vv) : ind(ii,jj), v(vv) {}
    Index ind;
    T v;
    bool operator==(Value const &rh) const { return ind==rh.ind;}
    bool operator<(Value const & rh) const { return ind<rh.ind;}
  };

}

template<typename T, int S>
class SparseMatrix {
public:
  typedef sparseMatrixDetails::Index Index;

  // slow accessors
  T operator()(int i, int j) const {
    return vs[find(i,j)];
  }
  T & operator()(int i, int j) {
    return vs[find(i,j)];
  }

  int find(int i, int j) const {
    Index ind(i,j);
    std::pair<Index const*,Index const *> ret = std::equal_range(indeces,indeces+S,ind);
    return (ret.first==ret.second) ? S : ret.first-indeces;
  }

  Index indeces[S];
  T vs[S+1]; // s+1 is 0;
};

template<typename T, int S>
class SparseMatrixBuilder {
public:
  typedef sparseMatrixDetails::Index Index;
  typedef sparseMatrixDetails::Value<T> Value;

  typedef  SparseMatrix<T,S> Mat;
  typedef  SparseMatrixBuilder<T,S> self;

  SparseMatrixBuilder(Mat & im): n(0), mat(im){}

  self & operator()(int i, int j, T v) {
    // ::assert(n<S);
    values[n] = Value(i,j,v);
    ++n;    
    if (n==S) finalize();
    return *this;
  }

void load(Mat const & m) {
    n=S;
    mat.vs[S]=T(0);
    for (int i=0; i!=S; ++i) {
      values[i].ind = m.indeces[i];
      values[i].v = m.vs[i];
    }
  }


  void transpose(Mat & im) {
    load(im);
    for (int i=0; i!=S; ++i)
      values[i].ind.transpose();
    finalize();
  }

  void finalize() {
    std::sort(values,values+S);
    mat.vs[S]=T(0);
    for (int i=0; i!=S; ++i) {
      mat.indeces[i]=values[i].ind;
      mat.vs[i]= values[i].v;
    }
  }
  
  int n;
  Mat & mat;
  Value values[S];
};
  
template<typename T, int N, int M, int S>
ROOT::Math::SVector<T,N> prod(SparseMatrix<T,S> const & m, ROOT::Math::SVector<T,M> const & v) {
  ROOT::Math::SVector<T,N> ret;
  for (int k=0; k!=S; ++k) {
    sparseMatrixDetails::Index const & ind = m.indeces[k];
    ret(ind.i) += m.vs[k]*v(ind.j); 
  }
  return ret;
}

template<typename T, int M, int N, int S>
ROOT::Math::SVector<T,N> prod(ROOT::Math::SVector<T,M> const & v, SparseMatrix<T,S> const & m) {
  ROOT::Math::SVector<T,N> ret;
  for (int k=0; k!=S; ++k) {
    sparseMatrixDetails::Index const & ind = m.indeces[k];
    ret(ind.j) += m.vs[k]*v(ind.i); 
  }
  return ret;
}

template<typename T, int N, int M, int S>
ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> >
similarity(SparseMatrix<T,S> const & m1,ROOT::Math::SMatrix<T,M,M,ROOT::Math::MatRepSym<T,M> > m2) {
  // ret(k,l) = m1(k,i)*m2(i,j)*m1(l,j)
  ROOT::Math::SMatrix<double,N,N,ROOT::Math::MatRepSym<double,N> > ret;
  for (int i=0; i!=S; ++i) {
    sparseMatrixDetails::Index const & ind1 = m1.indeces[i];
    for (int j=0; j!=S; ++j) {
      sparseMatrixDetails::Index const & ind2 = m1.indeces[j];
      if (ind2.i>ind1.i) break;   // ordered in i & we have to assign it ONCE!!!!
      ret(ind1.i,ind2.i) += m1.vs[i]*m2(ind1.j,ind2.j)*m1.vs[j];
    }
  }
  return ret;
}


template<typename T, int N, int M, int S>
ROOT::Math::SMatrix<T,N,N,ROOT::Math::MatRepSym<T,N> >
similarityT(SparseMatrix<T,S> const & m1,ROOT::Math::SMatrix<T,M,M,ROOT::Math::MatRepSym<T,M> > m2) {
  // ret(k,l) = m1(i,k)*m2(i,j)*m1(j,l)
  ROOT::Math::SMatrix<double,N,N,ROOT::Math::MatRepSym<double,N> > ret;

  for (int i=0; i!=S; ++i) {
    sparseMatrixDetails::Index const & ind1 = m1.indeces[i];
    for (int j=0; j!=S; ++j) {
      sparseMatrixDetails::Index const & ind2 = m1.indeces[j];
      if (ind2.j>ind1.j) continue;   // we have to assign it ONCE!!!!
      ret(ind1.j,ind2.j) += m1.vs[i]*m2(ind1.i,ind2.i)*m1.vs[j];

    }
  }
  /*
  unsigned int s = 0;  // l span storage of sym matrices
  // not optimized on the sparse matrix yet
  for(int k=0; k<N; ++k) {
    for(int l=0; l<=k; ++l) {
      for (int ik=0; ik!=S; ++ik) {
	sparseMatrixDetails::Index const & ind1 = m1.indeces[ik];
	if (ind1.j!=k) continue;
	for (int jl=0; jl!=S; ++jl) {
	  sparseMatrixDetails::Index const & ind2 = m1.indeces[jl];
	  if (ind2.j!=l) continue;
	  ret.fRep.Array()[s] += m1.vs[ik]*m2(ind1.i,ind2.i)*m1.vs[jl];
	}
      } 
      s++;
    }
  }
  */
  return ret;
}
#endif
