#include "SparseMatrix.h"
#include<iostream>

template<typename T, int S>
void print(SparseMatrix<T,S> const & m) {
  for (int i=0; i!=S; ++i)
    std::cout << m.indeces[i].i << "," << m.indeces[i].j << ":" << m.vs[i] << " ";
  std::cout << m.vs[S] << std::endl;
} 


int main() {

  typedef ROOT::Math::SVector<double,17> V17;
  typedef ROOT::Math::SVector<double,6> V6;

  typedef ROOT::Math::SMatrix<double,6,6,ROOT::Math::MatRepSym<double,6> > M6;
  typedef ROOT::Math::SMatrix<double,6,17> M617;
  typedef ROOT::Math::SMatrix<double,17,17,ROOT::Math::MatRepSym<double,17> > M17;

  typedef SparseMatrix<double,10> S;
  
  S s;
  S is1;
  S is2;
  SparseMatrixBuilder<double,10> b(s);
  SparseMatrixBuilder<double,10> bi1(is1);
  SparseMatrixBuilder<double,10> bi2(is2);

  b(0,1,1.)(1,0,-1.)(2,2,1.)(0,7,1.)(1,8,1.)(2,9,1.);
  b(4,4,1.)(4,5,1.)(5,4,-1.)(5,5,1.);

  bi1(1,0,1.)(0,1,-1.)(2,2,1.)(7,0,1.)(8,1,1.)(9,2,1.);
  bi1(4,4,1.)(5,4,1.)(4,5,-1.)(5,5,1.);

  bi2.transpose(s);

  print(s);
  print(is1);
  print(is2);

  M617 m617;
  // not to be  used in real code
  for (int i=0; i!=6; ++i)
    for (int j=0; j!=17; ++j)
      m617(i,j) = s(i,j);
  
  std::cout<< m617<<std::endl;

  V6 v6;
  for (int i=0; i!=6; ++i) 
    v6(i) = i;

  V17 v17a1 = prod<double,17,6,10>(is1,v6);

  V17 v17b1 = prod<double,6,17,10>(v6,s);

  std::cout << v17a1 << std::endl;
  std::cout << v17b1 << std::endl;

  V17 v17a2 = ROOT::Math::Transpose(m617)*v6;

  V17 v17b2 = v6*m617;

  std::cout << v17a2 << std::endl;
  std::cout << v17b2 << std::endl;
  if (v17a1!=v17a2) std::cout << "error in v17a" << std::endl;
  if (v17b1!=v17b2) std::cout << "error in v17b" << std::endl;


  M17 m17;
  for (int i=0; i!=17; ++i) 
    for (int j=0; j!=i; ++j) 
      m17(i,j) = i*20+j;

  M6 m6a1 = similarity<double,6,17,10>(s,m17);
  M6 m6a2 = ROOT::Math::Similarity(m617,m17);


  std::cout << m6a1 << std::endl;
  std::cout << m6a2 << std::endl;
  if (m6a1!=m6a2) std::cout << "error in m6a" << std::endl;

  M17 m17b1 = similarityT<double,17,6,10>(s,m6a1);
  M17 m17b2 = ROOT::Math::SimilarityT(m617,m6a1);

  M17 m17c1 = similarity<double,17,6,10>(is2,m6a1);



  if (m17b1!=m17b2) std::cout << "error in m17b" << std::endl;
  if (m17c1!=m17b2) std::cout << "error in m17c" << std::endl;


  return 0;

}
