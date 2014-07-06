#include <Math/SVector.h>
#include <Math/SMatrix.h>

// Use ".!" in VIM
// for I in $(seq 1 6); do echo "typedef ROOT::Math::SVector<double,$I> AlgebraicVector$I;"; done
typedef ROOT::Math::SVector<double,1> AlgebraicVector1;
typedef ROOT::Math::SVector<double,2> AlgebraicVector2;
typedef ROOT::Math::SVector<double,3> AlgebraicVector3;
typedef ROOT::Math::SVector<double,4> AlgebraicVector4;
typedef ROOT::Math::SVector<double,5> AlgebraicVector5;
typedef ROOT::Math::SVector<double,6> AlgebraicVector6;

// for I in $(seq 1 6); do echo "typedef ROOT::Math::SMatrix<double,$I,$I,ROOT::Math::MatRepSym<double,$I> > AlgebraicSymMatrix$I$I;"; done
typedef ROOT::Math::SMatrix<double,1,1,ROOT::Math::MatRepSym<double,1> > AlgebraicSymMatrix11;
typedef ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > AlgebraicSymMatrix22;
typedef ROOT::Math::SMatrix<double,3,3,ROOT::Math::MatRepSym<double,3> > AlgebraicSymMatrix33;
typedef ROOT::Math::SMatrix<double,4,4,ROOT::Math::MatRepSym<double,4> > AlgebraicSymMatrix44;
typedef ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepSym<double,5> > AlgebraicSymMatrix55;
typedef ROOT::Math::SMatrix<double,6,6,ROOT::Math::MatRepSym<double,6> > AlgebraicSymMatrix66;

// for I in $(seq 1 6); do for J in $(seq 1 6); do echo "typedef ROOT::Math::SMatrix<double,$I,$J,ROOT::Math::MatRepStd<double,$I,$J> > AlgebraicMatrix$I$J;"; done; done
typedef ROOT::Math::SMatrix<double,1,1,ROOT::Math::MatRepStd<double,1,1> > AlgebraicMatrix11;
typedef ROOT::Math::SMatrix<double,1,2,ROOT::Math::MatRepStd<double,1,2> > AlgebraicMatrix12;
typedef ROOT::Math::SMatrix<double,1,3,ROOT::Math::MatRepStd<double,1,3> > AlgebraicMatrix13;
typedef ROOT::Math::SMatrix<double,1,4,ROOT::Math::MatRepStd<double,1,4> > AlgebraicMatrix14;
typedef ROOT::Math::SMatrix<double,1,5,ROOT::Math::MatRepStd<double,1,5> > AlgebraicMatrix15;
typedef ROOT::Math::SMatrix<double,1,6,ROOT::Math::MatRepStd<double,1,6> > AlgebraicMatrix16;
typedef ROOT::Math::SMatrix<double,2,1,ROOT::Math::MatRepStd<double,2,1> > AlgebraicMatrix21;
typedef ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepStd<double,2,2> > AlgebraicMatrix22;
typedef ROOT::Math::SMatrix<double,2,3,ROOT::Math::MatRepStd<double,2,3> > AlgebraicMatrix23;
typedef ROOT::Math::SMatrix<double,2,4,ROOT::Math::MatRepStd<double,2,4> > AlgebraicMatrix24;
typedef ROOT::Math::SMatrix<double,2,5,ROOT::Math::MatRepStd<double,2,5> > AlgebraicMatrix25;
typedef ROOT::Math::SMatrix<double,2,6,ROOT::Math::MatRepStd<double,2,6> > AlgebraicMatrix26;
typedef ROOT::Math::SMatrix<double,3,1,ROOT::Math::MatRepStd<double,3,1> > AlgebraicMatrix31;
typedef ROOT::Math::SMatrix<double,3,2,ROOT::Math::MatRepStd<double,3,2> > AlgebraicMatrix32;
typedef ROOT::Math::SMatrix<double,3,3,ROOT::Math::MatRepStd<double,3,3> > AlgebraicMatrix33;
typedef ROOT::Math::SMatrix<double,3,4,ROOT::Math::MatRepStd<double,3,4> > AlgebraicMatrix34;
typedef ROOT::Math::SMatrix<double,3,5,ROOT::Math::MatRepStd<double,3,5> > AlgebraicMatrix35;
typedef ROOT::Math::SMatrix<double,3,6,ROOT::Math::MatRepStd<double,3,6> > AlgebraicMatrix36;
typedef ROOT::Math::SMatrix<double,4,1,ROOT::Math::MatRepStd<double,4,1> > AlgebraicMatrix41;
typedef ROOT::Math::SMatrix<double,4,2,ROOT::Math::MatRepStd<double,4,2> > AlgebraicMatrix42;
typedef ROOT::Math::SMatrix<double,4,3,ROOT::Math::MatRepStd<double,4,3> > AlgebraicMatrix43;
typedef ROOT::Math::SMatrix<double,4,4,ROOT::Math::MatRepStd<double,4,4> > AlgebraicMatrix44;
typedef ROOT::Math::SMatrix<double,4,5,ROOT::Math::MatRepStd<double,4,5> > AlgebraicMatrix45;
typedef ROOT::Math::SMatrix<double,4,6,ROOT::Math::MatRepStd<double,4,6> > AlgebraicMatrix46;
typedef ROOT::Math::SMatrix<double,5,1,ROOT::Math::MatRepStd<double,5,1> > AlgebraicMatrix51;
typedef ROOT::Math::SMatrix<double,5,2,ROOT::Math::MatRepStd<double,5,2> > AlgebraicMatrix52;
typedef ROOT::Math::SMatrix<double,5,3,ROOT::Math::MatRepStd<double,5,3> > AlgebraicMatrix53;
typedef ROOT::Math::SMatrix<double,5,4,ROOT::Math::MatRepStd<double,5,4> > AlgebraicMatrix54;
typedef ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepStd<double,5,5> > AlgebraicMatrix55;
typedef ROOT::Math::SMatrix<double,5,6,ROOT::Math::MatRepStd<double,5,6> > AlgebraicMatrix56;
typedef ROOT::Math::SMatrix<double,6,1,ROOT::Math::MatRepStd<double,6,1> > AlgebraicMatrix61;
typedef ROOT::Math::SMatrix<double,6,2,ROOT::Math::MatRepStd<double,6,2> > AlgebraicMatrix62;
typedef ROOT::Math::SMatrix<double,6,3,ROOT::Math::MatRepStd<double,6,3> > AlgebraicMatrix63;
typedef ROOT::Math::SMatrix<double,6,4,ROOT::Math::MatRepStd<double,6,4> > AlgebraicMatrix64;
typedef ROOT::Math::SMatrix<double,6,5,ROOT::Math::MatRepStd<double,6,5> > AlgebraicMatrix65;
typedef ROOT::Math::SMatrix<double,6,6,ROOT::Math::MatRepStd<double,6,6> > AlgebraicMatrix66;


/// ============= When we need templated root objects 
template <unsigned int D1, unsigned int D2=D1> struct AlgebraicROOTObject {
    typedef typename ROOT::Math::SVector<double,D1> Vector;
    typedef typename ROOT::Math::SMatrix<double,D1,D1,ROOT::Math::MatRepSym<double,D1> > SymMatrix;
    typedef typename ROOT::Math::SMatrix<double,D1,D2,ROOT::Math::MatRepStd<double,D1,D2> > Matrix;
};

typedef ROOT::Math::SMatrixIdentity AlgebraicMatrixID;


template<int D> 
void go() {
    
  typedef typename AlgebraicROOTObject<D,5>::Matrix MatD5;
  typedef typename AlgebraicROOTObject<5,D>::Matrix Mat5D;
  typedef typename AlgebraicROOTObject<D,D>::SymMatrix SMatDD;
  typedef typename AlgebraicROOTObject<D>::Vector VecD;

  /*
  double pzSign = tsos.localParameters().pzSign();

  MeasurementExtractor me(tsos);

  AlgebraicVector5 x = tsos.localParameters().vector();
  const AlgebraicSymMatrix55 &C = (tsos.localError().matrix());
  // Measurement matrix
  MatD5 H = asSMatrix<D,5>(aRecHit.projectionMatrix());

  // Residuals of aPredictedState w.r.t. aRecHit, 
  VecD r = asSVector<D>(aRecHit.parameters()) - me.measuredParameters<D>(aRecHit);

  // and covariance matrix of residuals
  SMatDD V = asSMatrix<D>(aRecHit.parametersError());
  SMatDD R = V + me.measuredError<D>(aRecHit);
  int ierr = ! R.Invert(); // if (ierr != 0) throw exception;
  //R.Invert();

  // Compute Kalman gain matrix
  Mat5D K = C * ROOT::Math::Transpose(H) * R;

  // Compute local filtered state vector
  AlgebraicVector5 fsv = x + K * r;
  */

  double aH[] =
    {0, 0, 0, 1, 0, 
     0, 0, 0, 0, 1};
  
  MatD5 H(aH,D*5);

  double aV[] =
    {1.7827e-05, 0.00016332, 0.0026591};

  SMatDD V(aV,3); 

  double aC[] = 
    {0.60194, -0.14262, 0.020447, 2.2361, -0.27993,
     -0.14262, 0.033806, -0.0044397, -0.52993, 0.059251, 
     0.020447, -0.0044397, 0.014513, 0.073261, -0.25078, 
     2.2361, -0.52993, 0.073261, 8.3081, -0.99288, 
     -0.27993, 0.059251, -0.25078, -0.99288, 4.3449};

  AlgebraicMatrix55 fC(aC,25);
  // AlgebraicSymMatrix55 C = fC.LowerBlock();
  AlgebraicSymMatrix55 C = fC.UpperBlock();
 
 
  double aR[] =
    {0.12374, 0.028255, 0.23647};
  
  SMatDD R(aR,3);

  double afsv[] = 
    {-0.16004, 0.034981, 0.22923, -0.15493, 2.6084};
  
  AlgebraicVector5 fsv(afsv,5);


    // Compute Kalman gain matrix
  Mat5D K = C * ROOT::Math::Transpose(H) * R;

  // Compute covariance matrix of local filtered state vector
  AlgebraicMatrix55 I = AlgebraicMatrixID();
  AlgebraicMatrix55 M = I - K * H;
  AlgebraicSymMatrix55 fse = ROOT::Math::Similarity(M, C) + ROOT::Math::Similarity(K, V);


  std::cout << fse << std::endl;

}

int main() {

  go<2>();
  return 0;
}
