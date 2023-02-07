#include <math.h>
#include "LU.h"
#ifdef VI_OPT
#include <x86intrin.h>
#include <float.h>
// #include "dmax.h"
#endif
double LU_num_flops(int N)
{
        /* rougly 2/3*N^3 */

    double Nd = (double) N;

    return (2.0 * Nd *Nd *Nd/ 3.0);
}


void LU_copy_matrix(int M, int N, double **lu, double **A)
{
    int i;
    int j;

    for (i=0; i<M; i++)
        for (j=0; j<N; j++)
            lu[i][j] = A[i][j];
}


int LU_factor(int M, int N, double **A,  int *pivot)
{
#ifdef VI_OPT
  unsigned long long const smask= ~0x8000000000000000LL;
  const unsigned long long smasks[4] = { smask, smask, smask, smask };
  const __m256d sign = _mm256_castsi256_pd(_mm256_loadu_si256((__m256i const *)smasks));
//    double col[M];
#endif
    int minMN =  M < N ? M : N;
    int j=0;

    for (j=0; j<minMN; j++)
    {
        /* find pivot in column j and  test for singularity. */

        int jp=j;
        int i;
        
#ifdef VI_OPT
//  slow
//       for (i=j; i!=M; ++i) col[i] = fabs(A[i][j]);
//       jp = dmax(col+j,col+M) - col;
       if (M-j < 4) {
#endif
//       for (i=j+1; i<M; i++)
//         jp = (fabs(A[i][j])>fabs(A[jp][j])) ? i : jp;

        double t = fabs(A[j][j]);
        for (i=j+1; i<M; i++) {
          double ab = fabs(A[i][j]);
          if ( ab > t)  {
             jp = i;
             t = ab;
          }
        }
#ifdef VI_OPT
        } else {

         int n = M-j;
         int off = 4-(n&3); // modulus 4
         if (4==off) off=0;
         double p[4];
         double k[4];  
         for (int i=0; i!=off;++i) {
           p[i] = DBL_MIN;
           k[i]=(i-off+j);
         }
         int jj=j;
         for (int i=off;i!=4;++i) {
           p[i] = A[jj][j];
           k[i]=jj++;
         }
         __m256d fmin = _mm256_and_pd(sign,_mm256_loadu_pd(p)); //abs
         __m256d res =  _mm256_loadu_pd(k) + _mm256_set1_pd(0.5);
  
         __m256d incr = _mm256_set1_pd(4.);
         __m256d ind = res;

         for (; jj!=M; jj+=4) {
          ind =  _mm256_add_pd(incr,ind);
          __m256d ftes = _mm256_setr_pd(A[jj][j],A[jj+1][j],A[jj+2][j],A[jj+3][j]);
          ftes = _mm256_and_pd(sign,ftes); //abs
          __m256d mask = _mm256_cmp_pd(ftes,fmin, _CMP_GT_OS);
          fmin = _mm256_max_pd(fmin,ftes);
          res =  _mm256_blendv_pd(res,ind,mask);
         }
         _mm256_storeu_pd(k,res);
         _mm256_storeu_pd(p,fmin);
         jp=k[0];
         for (int i=1; i!=4; ++i) {
           if (p[i]>p[0]) {
            p[0]=p[i];
            jp=k[i];
          } 
         }
         
  
        }
#endif        
        pivot[j] = jp;

        /* jp now has the index of maximum element  */
        /* of column j, below the diagonal          */

        if ( A[jp][j] == 0 )                 
            return 1;       /* factorization failed because of zero pivot */


        if (jp != j)
        {
            /* swap rows j and jp */
            double *tA = A[j];
            A[j] = A[jp];
            A[jp] = tA;
        }

        if (j<M-1)                /* compute elements j+1:M of jth column  */
        {
            /* note A(j,j), was A(jp,p) previously which was */
            /* guarranteed not to be zero (Label #1)         */

            double recp =  1.0 / A[j][j];
            int k;
            for (k=j+1; k<M; k++)
                A[k][j] *= recp;
        }


        if (j < minMN-1)
        {
            /* rank-1 update to trailing submatrix:   E = E - x*y; */
            /* E is the region A(j+1:M, j+1:N) */
            /* x is the column vector A(j+1:M,j) */
            /* y is row vector A(j,j+1:N)        */

            int ii;
            for (ii=j+1; ii<M; ii++)
            {
                double *Aii = A[ii];
                double *Aj = A[j];
                double AiiJ = Aii[j];
                int jj;
                for (jj=j+1; jj<N; jj++)
                  Aii[jj] -= AiiJ * Aj[jj];

            }
        }
    }

    return 0;
}

