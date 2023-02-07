#include "SOR.h"

    double SOR_num_flops(int M, int N, int num_iterations)
    {
        double Md = (double) M;
        double Nd = (double) N;
        double num_iterD = (double) num_iterations;

        return (Md-1)*(Nd-1)*num_iterD*6.0;
    }

    void SOR_execute(int M, int N, double omega, double **g, int 
            num_iterations)
    {

//        double ** G = (double**)(__builtin_assume_aligned(g,16));
        double ** G = g; 
        double omega_over_four = omega * 0.25;
        double one_minus_omega = 1.0 - omega;

        /* update interior points */

        int Mm1 = M-1;
        int Nm1 = N-1; 
        int p;
        int i;
        int j;
        double * __restrict__ Gi;
        double * __restrict__ Gim1;
        double * __restrict__ Gip1;

        for (p=0; p<num_iterations; p++)
        {
            for (i=1; i<Mm1; i++)
            {
//                Gi = (double*)(__builtin_assume_aligned(G[i],16);
//               Gim1 = (double*)(__builtin_assume_aligned(G[i-1],16);
//               Gip1 = (double*)(__builtin_assume_aligned(G[i+1],16);
             	Gi = G[i];
                Gim1 = G[i-1];
                Gip1 = G[i+1];
                for (j=1; j<Nm1; j++)
                    Gi[j] = omega_over_four * (Gim1[j] + Gip1[j] + Gi[j-1] 
                                + Gi[j+1]) + one_minus_omega * Gi[j];
            }
        }
    }
            
