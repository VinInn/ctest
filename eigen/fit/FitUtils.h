#ifndef RecoPixelVertexing_PixelTrackFitting_interface_FitUtils_h
#define RecoPixelVertexing_PixelTrackFitting_interface_FitUtils_h



#include "FitResult.h"

#include "../choleskyInversion.h"


namespace Rfit
{


  constexpr Float d = 1.e-4;          //!< used in numerical derivative (J2 in Circle_fit())

#ifdef __CUDA_ARCH__
  __device__ __constant__ Float one = 1.0;
  __device__ __constant__ Float two = 2.0;
  __device__ __constant__ Float op5 = 0.5;
#else
  constexpr Float one = 1.0;
  constexpr Float two = 2.0;
  constexpr Float op5 = 0.5;
#endif


  using VectorXd = VectorXFF;
  using MatrixXd = MatrixXFF;
  template<int N>
  using MatrixNd = Eigen::Matrix<FF, N, N>;
  template<int N>
  using MatrixNplusONEd = Eigen::Matrix<FF, N+1, N+1>; 
  template<int N>
  using ArrayNd = Eigen::Array<FF, N, N>;
  template<int N>
  using Matrix2Nd = Eigen::Matrix<FF, 2 * N, 2 * N>;
  template<int N>
  using Matrix3Nd = Eigen::Matrix<FF, 3 * N, 3 * N>;
  template<int N>
  using Matrix2xNd = Eigen::Matrix<FF, 2, N>;
  template<int N>
  using Array2xNd = Eigen::Array<FF, 2, N>;
  template<int N>
  using MatrixNx3d = Eigen::Matrix<FF, N, 3>;
  template<int N>
  using MatrixNx5d = Eigen::Matrix<FF, N, 5>;
  template<int N>
  using VectorNd = Eigen::Matrix<FF, N, 1>;
  template<int N>
  using VectorNplusONEd  = Eigen::Matrix<FF, N+1, 1>;
  template<int N>
  using Vector2Nd = Eigen::Matrix<FF, 2 * N, 1>;
  template<int N>
  using Vector3Nd = Eigen::Matrix<FF, 3 * N, 1>;
  template<int N>
  using RowVectorNd = Eigen::Matrix<FF, 1, 1, N>;
  template<int N>
  using RowVector2Nd = Eigen::Matrix<FF, 1, 2 * N>;


  using Matrix2x3d = Eigen::Matrix<FF, 2, 3>;

  
  using Matrix3f = Eigen::Matrix3f;
  using Vector3f = Eigen::Vector3f;
  using Vector4f = Eigen::Vector4f;
  using Vector6f = Eigen::Matrix<FF, 6, 1>;

  
  using u_int = unsigned int;

// Eigen interface
  template<typename V>
  __host__ __device__
  inline constexpr
  auto squaredNorm(V const & src) {
#ifdef FAST_SN 
     return squaredNorm(src,V::SizeAtCompileTime);
#else
    return src.squaredNorm();
#endif
  }

  
  template <class C>
  __host__ __device__ void printIt(C* m, const char* prefix = "")
  {
#ifdef RFIT_DEBUG
    for (u_int r = 0; r < m->rows(); ++r)
      {
        for (u_int c = 0; c < m->cols(); ++c)
	  {
            printf("%s Matrix(%d,%d) = %g\n", prefix, r, c, (*m)(r, c));
	  }
      }
#endif
  }
  
  /*!
    \brief raise to square.
  */
  template <typename T>
  constexpr T sqr(const T a)
  {
    return square(a);
  }
  
  /*!
    \brief Compute cross product of two 2D vector (assuming z component 0),
    returning z component of the result.
    \param a first 2D vector in the product.
    \param b second 2D vector in the product.
    \return z component of the cross product.
  */
  
  __host__ __device__ inline FF cross2D(const Vector2d& a, const Vector2d& b)
  {
    return a.x() * b.y() - a.y() * b.x();
  }
  
  /*!
   *  load error in CMSSW format to our formalism
   *  
   */
  template<typename M6xNf, typename M2Nd>
  __host__ __device__ void loadCovariance2D(M6xNf const & ge,  M2Nd & hits_cov) {
    // Index numerology:
    // i: index of the hits/point (0,..,3)
    // j: index of space component (x,y,z)
    // l: index of space components (x,y,z)
    // ge is always in sync with the index i and is formatted as:
    // ge[] ==> [xx, xy, yy, xz, yz, zz]
    // in (j,l) notation, we have:
    // ge[] ==> [(0,0), (0,1), (1,1), (0,2), (1,2), (2,2)]
    // so the index ge_idx corresponds to the matrix elements:
    // | 0  1  3 |
    // | 1  2  4 |
    // | 3  4  5 |
    constexpr uint32_t hits_in_fit = M6xNf::ColsAtCompileTime;
    for (uint32_t i=0; i< hits_in_fit; ++i) {
      auto ge_idx = 0; auto j=0; auto l=0;
      hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      ge_idx = 2; j=1; l=1;
      hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      ge_idx = 1; j=1; l=0;
      hits_cov(i + l * hits_in_fit, i + j * hits_in_fit) =
      hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
    }
  }
  
  template<typename M6xNf, typename M3xNd>
  __host__ __device__ void loadCovariance(M6xNf const & ge,  M3xNd & hits_cov) {
    
    // Index numerology:
    // i: index of the hits/point (0,..,3)
    // j: index of space component (x,y,z)
    // l: index of space components (x,y,z)
    // ge is always in sync with the index i and is formatted as:
    // ge[] ==> [xx, xy, yy, xz, yz, zz]
    // in (j,l) notation, we have:
    // ge[] ==> [(0,0), (0,1), (1,1), (0,2), (1,2), (2,2)]
    // so the index ge_idx corresponds to the matrix elements:
    // | 0  1  3 |
    // | 1  2  4 |
    // | 3  4  5 |
    constexpr uint32_t hits_in_fit = M6xNf::ColsAtCompileTime;
    for (uint32_t i=0; i<hits_in_fit; ++i) {
      auto ge_idx = 0; auto j=0; auto l=0;
      hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      ge_idx = 2; j=1; l=1;
      hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      ge_idx = 5; j=2; l=2;
      hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      ge_idx = 1; j=1; l=0;
      hits_cov(i + l * hits_in_fit, i + j * hits_in_fit) =
	hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      ge_idx = 3; j=2; l=0;
      hits_cov(i + l * hits_in_fit, i + j * hits_in_fit) =
	hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
      ge_idx = 4; j=2; l=1;
      hits_cov(i + l * hits_in_fit, i + j * hits_in_fit) =
	hits_cov(i + j * hits_in_fit, i + l * hits_in_fit) = ge.col(i)[ge_idx];
    }
  }

  /*!
    \brief Transform circle parameter from (X0,Y0,R) to (phi,Tip,p_t) and
    consequently covariance matrix.
    \param circle_uvr parameter (X0,Y0,R), covariance matrix to
    be transformed and particle charge.
    \param B magnetic field in Gev/cm/c unit.
    \param error flag for errors computation.
  */
  __host__ __device__
  inline void par_uvrtopak(circle_fit& circle, const Float B, const bool error)
  {
    Float fq = float(circle.q);
    Vector3d par_pak;
    const FF temp0 = squaredNorm(circle.par.head(2));
    const FF temp1 = sqrt(temp0);
    par_pak << std::atan2(fq * toSingle(circle.par(0)), -fq * toSingle(circle.par(1))),
      fq * (temp1 - circle.par(2)), circle.par(2) * B;
    if (error)
      {
        const FF temp2 = sqr(circle.par(0))  / temp0;
        const FF temp3 = fq / temp1;
        Matrix3d J4;
        J4 << -circle.par(1) * temp2 / sqr(circle.par(0)), temp2/ circle.par(0), 0., 
	  circle.par(0) * temp3, circle.par(1) * temp3, -fq,
	  0., 0., B;
        circle.cov = J4 * circle.cov * J4.transpose();
      }
    circle.par = par_pak;
  }
  
}



#endif
