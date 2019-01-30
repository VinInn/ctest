#ifndef RecoPixelVertexing_PixelTrackFitting_interface_FitUtils_h
#define RecoPixelVertexing_PixelTrackFitting_interface_FitUtils_h


#include "FitResult.h"


namespace Rfit
{
  
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
  __host__ __device__ inline T sqr(const T a)
  {
    return a * a;
  }
  
  /*!
    \brief Compute cross product of two 2D vector (assuming z component 0),
    returning z component of the result.
    \param a first 2D vector in the product.
    \param b second 2D vector in the product.
    \return z component of the cross product.
  */
  
  __host__ __device__ inline double cross2D(const Vector2d& a, const Vector2d& b)
  {
    return a.x() * b.y() - a.y() * b.x();
  }
  
  /*!
   *  load error in CMSSW format to our formalism
   *  
   */
  template<typename M6x4f, typename M2Nd>
  __host__ __device__ void loadCovariance2D(M6x4f const & ge,  M2Nd & hits_cov) {
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
    constexpr uint32_t  hits_in_fit = 4; // Fixme
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
  
  template<typename M6x4f, typename M3xNd>
  __host__ __device__ void loadCovariance(M6x4f const & ge,  M3xNd & hits_cov) {
    
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
    constexpr uint32_t  hits_in_fit = 4; // Fixme
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
  
}


#endif
