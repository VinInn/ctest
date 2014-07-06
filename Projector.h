#ifndef Projector_H
#define Projector_H

#include <Math/SVector.h>
#include <Math/SMatrix.h>
#include <algorithm>

/*
template<size_t D, size_t D1, size_t D2, typename R>
struct Projected {};

template<size_t D, size_t D1, size_t D2>
struct Projected<D, D1, D2, MatRepStd<double, D1, D2> > {
  typedef Root::Math::SMatrix<double,D, D1, MatRepStd<double, D, D1> > LeftProjected;
  typedef Root::Math::SMatrix<double,D2, D, MatRepStd<double, D2, D> > RightProjected;

};

template<size_t D, size_t D1, size_t D2=D1>
struct Projected<D, D1, D2, MatRepSym<double, D1> > {
  typedef Root::Math::SMatrix<double,D, D1, MatRepSym<double, D, D1> > LeftProjected;
  typedef Root::Math::SMatrix<double,D1, D, MatRepSym<double, D1, D> > RightProjected;

};
*/


template<size_t D>
class Projector {
public:
  typedef ROOT::Math::SMatrix<double, D, D, ROOT::Math::MatRepSym<double, D> > Matrix;
  typedef ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5> > Matrix55;
  typedef ROOT::Math::SVector<double,D> Vector;
  typedef ROOT::Math::SVector<double,5> Vector5;

  typedef typename ROOT::Math::SMatrix<double,D,5,ROOT::Math::MatRepStd<double,D,5> > MatD5;
  typedef typename ROOT::Math::SMatrix<double,5,D,ROOT::Math::MatRepStd<double,5,D> > Mat5D;

  virtual Matrix project(Matrix55 const & m) const =0;
  virtual Vector project(Vector5 const & v) const =0;


};

template<size_t D>
inline typename Projector<D>::Matrix 
Similarity(Projector<D> const & p, 
	   typename Projector<D>::Matrix55 const & m) {
  return p.project(m);
}
/*
Projector<2>::Matrix 
Similarity(Projector<2> const & p, 
	   Projector<2>::Matrix55 const & m) {
  return p.project(m);
}
*/

template<size_t D>
class ProjectorBySubMatrix : public Projector<D> {
public:
  typedef Projector<D> super;
  typedef typename super::Matrix Matrix;
  typedef typename super::Matrix55 Matrix55;
  typedef typename super::Vector Vector;
  typedef typename super::Vector5 Vector5;
  typedef typename super::MatD5 MatD5;
  typedef typename super::Mat5D Mat5D;
 
  ProjectorBySubMatrix(size_t ifirstIndex) : m_firstIndex(ifirstIndex) {}

  Matrix project(Matrix55 const & m) const {
    return m.template Sub<Matrix>(m_firstIndex,m_firstIndex);
  } 

  Vector project(Vector5 const & v) const {
    return v.template Sub<Vector>(m_firstIndex);
  } 

private: 
  size_t m_firstIndex;

};


template<size_t D>
class ProjectorByGSlice : public Projector<D> {
public:
  typedef Projector<D> super;
  typedef typename super::Matrix Matrix;
  typedef typename super::Matrix55 Matrix55;
  typedef typename super::Vector Vector;
  typedef typename super::Vector5 Vector5;
  typedef typename super::MatD5 MatD5;
  typedef typename super::Mat5D Mat5D;
 
 ProjectorByGSlice(size_t iIndices[D]) {
   std::copy(iIndices, iIndices+D,m_Indices);
  }

  Matrix project(Matrix55 const & m) const {
    Matrix r;
    for (size_t i=0;i<D;++i)
      for (size_t j=i;j<D;++j)
	r(i,j)=m(m_Indices[i],m_Indices[j]);
    return r;
  } 

  Vector project(Vector5 const & v) const {
    Vector r;
    for (size_t i=0;i<D;++i)
      r(i)=v(m_Indices[i]);
    return r;
  } 

private: 

  size_t m_Indices[D]; 

};

template<size_t D>
class ProjectorBySimilarity : public Projector<D> {
public:
  typedef Projector<D> super;
  typedef typename super::MatD5 MatD5;
  typedef typename super::Mat5D Mat5D;
  typedef typename super::Matrix Matrix;
  typedef typename super::Matrix55 Matrix55;
  typedef typename super::Vector Vector;
  typedef typename super::Vector5 Vector5;
 
  ProjectorBySimilarity(MatD5 const & iH) : m_H(iH) {}

  Matrix project(Matrix55 const & m) const {
    return ROOT::Math::Similarity(m_H,m);
  } 

  Vector project(Vector5 const & v) const {
    return m_H*v;
  } 

private: 

  MatD5 m_H;

};








#endif //  Projector_H
