#pragma once

// imported form Smatrix


#include <cstddef>
#include <utility>
#include <type_traits>
#include <array>


/**
\defgroup MatrixStd Standard Matrix representation
\ingroup Matrix

Standard Matrix representation for a general D1 x D2 matrix.
This class is itself a template on the contained type T, the number of rows and the number of columns.
Its data member is an array T[nrows*ncols] containing the matrix data.
The data are stored in the row-major C convention.
For example, for a matrix, M, of size 3x3, the data \f$ \left[a_0,a_1,a_2,.......,a_7,a_8 \right] \f$d
are stored in the following order:

\f[
M = \left( \begin{array}{ccc}
a_0 & a_1 & a_2  \\
a_3 & a_4  & a_5  \\
a_6 & a_7  & a_8   \end{array} \right)
\f]

*/


   template <class T, unsigned int D1, unsigned int D2=D1>
   class MatrixStd {

   public:

      typedef T  value_type;

      inline constexpr const T& operator()(unsigned int i, unsigned int j) const {
         return fArray[i*D2+j];
      }
      inline constexpr T& operator()(unsigned int i, unsigned int j) {
         return fArray[i*D2+j];
      }
      inline constexpr T& operator[](unsigned int i) { return fArray[i]; }

      inline constexpr const T& operator[](unsigned int i) const { return fArray[i]; }

      inline constexpr T apply(unsigned int i) const { return fArray[i]; }

      inline constexpr T* Array() { return fArray; }

      inline constexpr const T* Array() const { return fArray; }

      template <class R>
      inline constexpr MatrixStd<T, D1, D2>& operator+=(const R& rhs) {
         for(unsigned int i=0; i<kSize; ++i) fArray[i] += rhs[i];
         return *this;
      }

      template <class R>
      inline constexpr MatrixStd<T, D1, D2>& operator-=(const R& rhs) {
         for(unsigned int i=0; i<kSize; ++i) fArray[i] -= rhs[i];
         return *this;
      }

      template <class R>
      inline constexpr MatrixStd<T, D1, D2>& operator=(const R& rhs) {
         for(unsigned int i=0; i<kSize; ++i) fArray[i] = rhs[i];
         return *this;
      }

      template <class R>
      inline constexpr bool operator==(const R& rhs) const {
         bool rc = true;
         for(unsigned int i=0; i<kSize; ++i) {
            rc = rc && (fArray[i] == rhs[i]);
         }
         return rc;
      }

      enum {
         /// return no. of matrix rows
         kRows = D1,
         /// return no. of matrix columns
         kCols = D2,
         /// return no of elements: rows*columns
         kSize = D1*D2
      };

   private:
      //T __attribute__ ((aligned (16))) fArray[kSize];
      T  fArray[kSize];
   };


//     template<unsigned int D>
//     struct Creator {
//       static const RowOffsets<D> & Offsets() {
//          static RowOffsets<D> off;
//           return off;
//       }

   /**
      Static structure to keep the conversion from (i,j) to offsets in the storage data for a
      symmetric matrix
   */

   template<unsigned int D>
   struct RowOffsets {
      inline constexpr RowOffsets() {
         int v[D];
         v[0]=0;
         for (unsigned int i=1; i<D; ++i)
            v[i]=v[i-1]+i;
         for (unsigned int i=0; i<D; ++i) {
            for (unsigned int j=0; j<=i; ++j)
               fOff[i*D+j] = v[i]+j;
            for (unsigned int j=i+1; j<D; ++j)
               fOff[i*D+j] = v[j]+i ;
         }
      }
      inline constexpr int operator()(unsigned int i, unsigned int j) const { return fOff[i*D+j]; }
      inline constexpr int apply(unsigned int i) const { return fOff[i]; }
      int fOff[D*D];
   };

  namespace rowOffsetsUtils {

    ///////////
    // Some meta template stuff
    template<int...> struct indices{};

    template<int I, class IndexTuple, int N>
    struct make_indices_impl;

    template<int I, int... Indices, int N>
    struct make_indices_impl<I, indices<Indices...>, N>
    {
      typedef typename make_indices_impl<I + 1, indices<Indices..., I>,
                     N>::type type;
    };

    template<int N, int... Indices>
    struct make_indices_impl<N, indices<Indices...>, N> {
      typedef indices<Indices...> type;
    };

    template<int N>
    struct make_indices : make_indices_impl<0, indices<>, N> {};
    // end of stuff



    template<int I0, class F, int... I>
    constexpr std::array<decltype(std::declval<F>()(std::declval<int>())), sizeof...(I)>
    do_make(F f, indices<I...>)
    {
      return  std::array<decltype(std::declval<F>()(std::declval<int>())),
             sizeof...(I)>{{ f(I0 + I)... }};
    }

    template<int N, int I0 = 0, class F>
    constexpr std::array<decltype(std::declval<F>()(std::declval<int>())), N>
    make(F f) {
      return do_make<I0>(f, typename make_indices<N>::type());
    }

  } // namespace rowOffsetsUtils


//_________________________________________________________________________________
   /**
      MatrixSym
      Matrix storage representation for a symmetric matrix of dimension NxN
      This class is a template on the contained type and on the symmetric matrix size, N.
      It has as data member an array of type T of size N*(N+1)/2,
      containing the lower diagonal block of the matrix.
      The order follows the lower diagonal block, still in a row-major convention.
      For example for a symmetric 3x3 matrix the order of the 6 elements
      \f$ \left[a_0,a_1.....a_5 \right]\f$ is:
      \f[
      M = \left( \begin{array}{ccc}
      a_0 & a_1  & a_3  \\
      a_1 & a_2  & a_4  \\
      a_3 & a_4 & a_5   \end{array} \right)
      \f]

      @ingroup Matrix
   */
   template <class T, unsigned int D>
   class MatrixSym {

   public:

#ifdef __NVCC__
    __device__ __host__ 
#endif
    inline constexpr MatrixSym(){}

    typedef T  value_type;


    inline constexpr T & operator()(unsigned int i, unsigned int j)
     { return fArray[offset(i, j)]; }

     inline constexpr  T const & operator()(unsigned int i, unsigned int j) const
     { return fArray[offset(i, j)]; }

     inline constexpr T& operator[](unsigned int i) {
       return fArray[off(i)];
     }

     inline constexpr  T const & operator[](unsigned int i) const {
       return fArray[off(i)];
     }

     inline constexpr  T apply(unsigned int i) const {
       return fArray[off(i)];
     }

     inline constexpr T* Array() { return fArray; }

     inline constexpr const T* Array() const { return fArray; }

      /**
         assignment : only symmetric to symmetric allowed
       */
      /*
      template <class R>
      inline constexpr MatrixSym<T, D>& operator=(const R&) {
         static_assert(0==1,
                      "Cannot_assign_general_to_symmetric_matrix_representation");
         return *this;
      }
      */
#ifdef __NVCC__
     __device__ __host__
#endif
      inline constexpr MatrixSym(const MatrixSym& rhs) {
         for(unsigned int i=0; i<kSize; ++i) fArray[i] = rhs.Array()[i];
      }
      inline constexpr MatrixSym<T, D>& operator=(const MatrixSym& rhs) {
         for(unsigned int i=0; i<kSize; ++i) fArray[i] = rhs.Array()[i];
         return *this;
      }

      /**
         self addition : only symmetric to symmetric allowed
       */
      /*
      template <class R>
      inline constexpr MatrixSym<T, D>& operator+=(const R&) {
         static_assert(0==1,
                      "Cannot_add_general_to_symmetric_matrix_representation");
         return *this;
      }
      */
      inline constexpr MatrixSym<T, D>& operator+=(const MatrixSym& rhs) {
         for(unsigned int i=0; i<kSize; ++i) fArray[i] += rhs.Array()[i];
         return *this;
      }

      /**
         self subtraction : only symmetric to symmetric allowed
       */
      /*
      template <class R>
      inline constexpr MatrixSym<T, D>& operator-=(const R&) {
         static_assert(0==1,
                      "Cannot_substract_general_to_symmetric_matrix_representation");
         return *this;
      }
      */
      inline constexpr MatrixSym<T, D>& operator-=(const MatrixSym& rhs) {
         for(unsigned int i=0; i<kSize; ++i) fArray[i] -= rhs.Array()[i];
         return *this;
      }
      template <class R>
      inline constexpr bool operator==(const R& rhs) const {
         bool rc = true;
         for(unsigned int i=0; i<D*D; ++i) {
            rc = rc && (operator[](i) == rhs[i]);
         }
         return rc;
      }

      enum {
         /// return no. of matrix rows
         kRows = D,
         /// return no. of matrix columns
         kCols = D,
         /// return no of elements: rows*columns
         kSize = D*(D+1)/2
      };

     static constexpr int off0(int i) { return i==0 ? 0 : off0(i-1)+i;}
     static constexpr int off2(int i, int j) { return j<i ? off0(i)+j : off0(j)+i; }
     static constexpr int off1(int i) { return off2(i/D, i%D);}

     static int off(int i) {
       static constexpr auto v = rowOffsetsUtils::make<D*D>(off1);
       return v[i];
     }

     static inline constexpr unsigned int
     offset(unsigned int i, unsigned int j)
     {
       if (j > i) std::swap(i, j);
       //return off(i*D+j);
       return (i>j) ? (i * (i+1) / 2) + j :  (j * (j+1) / 2) + i;
     }

   private:
      //T __attribute__ ((aligned (16))) fArray[kSize];
      T fArray[kSize];
   };


// here for test
    template <typename M1, typename M2>
#ifdef __NVCC__
     __device__ __host__
#endif
    inline constexpr void __attribute__((always_inline)) invert55(M1 const& src, M2& dst) {
      using F = typename std::remove_reference<decltype(src(0, 0))>::type;;
//#ifdef MNORM
      int e[5];
      for (int i=0; i<5; ++i) {
        std::frexp(src(i,i), &e[i]);
        e[i] =0; // /= 2;
      }
//#endif
      F one(1.0f);
      auto luc0 = one / std::ldexp(src(0, 0),-e[0]-e[0]);
      auto luc1 = std::ldexp(src(1, 0),-e[1]-e[0]);
      auto luc2 = std::ldexp(src(1, 1),-e[1]-e[1]) - luc0 * luc1 * luc1;
      luc2 = one / luc2;
      auto luc3 = std::ldexp(src(2, 0),-e[2]-e[0]);
      auto luc4 = std::ldexp(src(2, 1),-e[2]-e[1]) - (luc0 * luc1 * luc3);
      auto luc5 = std::ldexp(src(2, 2),-e[2]-e[2]) - (luc0 * luc3 * luc3 + luc2 * luc4 * luc4);
      luc5 = one / luc5;
      auto luc6 = std::ldexp(src(3, 0),-e[3]-e[0]);
      auto luc7 = std::ldexp(src(3, 1),-e[3]-e[1]) - luc0 * luc1 * luc6;
      auto luc8 = std::ldexp(src(3, 2),-e[3]-e[2]) - luc0 * luc3 * luc6 - luc2 * luc4 * luc7;
      auto luc9 = std::ldexp(src(3, 3),-e[3]-e[3]) - (luc0 * luc6 * luc6 + luc2 * luc7 * luc7 + luc8 * (luc8 * luc5));
      luc9 = one / luc9;
      auto luc10 = std::ldexp(src(4, 0),-e[4]-e[0]);
      auto luc11 = std::ldexp(src(4, 1),-e[4]-e[1]) - luc0 * luc1 * luc10;
      auto luc12 = std::ldexp(src(4, 2),-e[4]-e[2]) - luc0 * luc3 * luc10 - luc2 * luc4 * luc11;
      auto luc13 = std::ldexp(src(4, 3),-e[4]-e[3]) - luc0 * luc6 * luc10 - luc2 * luc7 * luc11 - luc5 * luc8 * luc12;
      auto luc14 =
          std::ldexp(src(4, 4),-e[4]-e[4]) - (luc0 * luc10 * luc10 + luc2 * luc11 * luc11 + luc5 * luc12 * luc12 + luc9 * luc13 * luc13);
      luc14 = one / luc14;

      auto li21 = -luc1 * luc0;
      auto li32 = -luc2 * luc4;
      auto li31 = (luc1 * (luc2 * luc4) - luc3) * luc0;
      auto li43 = -(luc8 * luc5);
      auto li42 = (luc4 * luc8 * luc5 - luc7) * luc2;
      auto li41 = (-luc1 * (luc2 * luc4) * (luc8 * luc5) + luc1 * (luc2 * luc7) + luc3 * (luc8 * luc5) - luc6) * luc0;
      auto li54 = -luc13 * luc9;
      auto li53 = (luc13 * luc8 * luc9 - luc12) * luc5;
      auto li52 = (-luc4 * luc8 * luc13 * luc5 * luc9 + luc4 * luc12 * luc5 + luc7 * luc13 * luc9 - luc11) * luc2;
      auto li51 = (luc1 * luc4 * luc8 * luc13 * luc2 * luc5 * luc9 - luc13 * luc8 * luc3 * luc9 * luc5 -
                   luc12 * luc4 * luc1 * luc2 * luc5 - luc13 * luc7 * luc1 * luc9 * luc2 + luc11 * luc1 * luc2 +
                   luc12 * luc3 * luc5 + luc13 * luc6 * luc9 - luc10) *
                  luc0;

      dst(0, 0) = std::ldexp(luc14 * li51 * li51 + luc9 * li41 * li41 + luc5 * li31 * li31 + luc2 * li21 * li21 + luc0,-e[0]-e[0]);
      dst(1, 0) = std::ldexp(luc14 * li51 * li52 + luc9 * li41 * li42 + luc5 * li31 * li32 + luc2 * li21,-e[1]-e[0]);
      dst(1, 1) = std::ldexp(luc14 * li52 * li52 + luc9 * li42 * li42 + luc5 * li32 * li32 + luc2,-e[1]-e[1]);
      dst(2, 0) = std::ldexp(luc14 * li51 * li53 + luc9 * li41 * li43 + luc5 * li31,-e[2]-e[0]);
      dst(2, 1) = std::ldexp(luc14 * li52 * li53 + luc9 * li42 * li43 + luc5 * li32,-e[2]-e[1]);
      dst(2, 2) = std::ldexp(luc14 * li53 * li53 + luc9 * li43 * li43 + luc5,-e[2]-e[2]);
      dst(3, 0) = std::ldexp(luc14 * li51 * li54 + luc9 * li41,-e[3]-e[0]);
      dst(3, 1) = std::ldexp(luc14 * li52 * li54 + luc9 * li42,-e[3]-e[1]);
      dst(3, 2) = std::ldexp(luc14 * li53 * li54 + luc9 * li43,-e[3]-e[2]);
      dst(3, 3) = std::ldexp(luc14 * li54 * li54 + luc9,-e[3]-e[3]);
      dst(4, 0) = std::ldexp(luc14 * li51,-e[4]-e[0]);
      dst(4, 1) = std::ldexp(luc14 * li52,-e[4]-e[1]);
      dst(4, 2) = std::ldexp(luc14 * li53,-e[4]-e[2]);
      dst(4, 3) = std::ldexp(luc14 * li54,-e[4]-e[3]);
      dst(4, 4) = std::ldexp(luc14,-e[4]-e[4]);
    }

