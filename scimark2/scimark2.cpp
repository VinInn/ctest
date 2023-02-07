#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Random.h"
#include "kernel.h"
#include "constants.h"

void print_banner(void);

int main(int argc, char *argv[])
{
  /* default to the (small) cache-contained version */
  
  double min_time = RESOLUTION_DEFAULT;
  int perm =0;
  int tail=-1;

  int FFT_size = FFT_SIZE;
  int SOR_size =  SOR_SIZE;
  int Sparse_size_M = SPARSE_SIZE_M;
  int Sparse_size_nz = SPARSE_SIZE_nz;
  int LU_size = LU_SIZE;
  
  
  /* run the benchmark */
  
  double res[6] = {0.0};
  Random R = new_Random_seed(RANDOM_SEED);
  
  
  if (argc > 1)
    {
      int current_arg = 1;
      
      if (strcmp(argv[1], "-help")==0  ||
	  strcmp(argv[1], "-h") == 0)
	{
	  fprintf(stderr, "Usage: [-large] [minimum_time] [permutation] [tail]\n");
	  exit(0);
	}
      
      if (strcmp(argv[1], "-large")==0)
	{
	  FFT_size = LG_FFT_SIZE/16;
	  SOR_size = LG_SOR_SIZE;
	  Sparse_size_M = LG_SPARSE_SIZE_M;
	  Sparse_size_nz = LG_SPARSE_SIZE_nz;
	  LU_size = LG_LU_SIZE;
	  
	  current_arg++;
	}
      else if (strcmp(argv[1], "-verylarge")==0)
	{
	  FFT_size = LG_FFT_SIZE;
	  SOR_size = 10*LG_SOR_SIZE;
	  Sparse_size_M = 10*LG_SPARSE_SIZE_M;
	  Sparse_size_nz = 10*LG_SPARSE_SIZE_nz;
	  LU_size = 10*LG_LU_SIZE;
	  
	  current_arg++;
	}
      
      if (current_arg < argc)
	{
	  min_time = atof(argv[current_arg]);
	  current_arg++;
	}
      
      if (current_arg < argc)
	{
	  perm = atoi(argv[current_arg]);
          current_arg++;
	}
      if (current_arg < argc)
        {
          tail = atoi(argv[current_arg]);
          current_arg++;
        }

    }
  
      
  print_banner();
  printf("Using %10.2f seconds min time per kenel.\n", min_time);
  printf("running permutation %d\n",perm);
  printf("running tail %d\n",tail);

  // heat-up
  double lll =   kernel_measureLU( LU_size, 2, R);  
  printf("LU first round %8.2f\n",lll);

  /* very trivial permutation trick */
  int i=perm;
  if (i>=0)
  for(;  i<perm+5; ++i) {
    switch (i%5) {
      case 0: 
	res[1] = kernel_measureFFT( FFT_size, min_time, R); 
	break;
      case 1:
	res[2] = kernel_measureSOR( SOR_size, min_time, R);   
	break;
      case 2:
	res[3] = kernel_measureMonteCarlo(min_time, R); 
	break;
      case 3:
	res[4] = kernel_measureSparseMatMult( Sparse_size_M, 
					      Sparse_size_nz, min_time, R);           
	break;
      case 4:
	res[5] = kernel_measureLU( LU_size, min_time, R);  
	break;
      }
  }  
  if (tail>=0) {
    // single benchmark or tail
    switch (tail%5) {
      case 0: 
	lll = kernel_measureFFT( FFT_size, min_time, R); 
	break;
      case 1:
	lll = kernel_measureSOR( SOR_size, min_time, R);   
	break;
      case 2:
	lll = kernel_measureMonteCarlo(min_time, R); 
	break;
      case 3:
	lll = kernel_measureSparseMatMult( Sparse_size_M, 
					      Sparse_size_nz, min_time, R);           
	break;
      case 4:
	lll = kernel_measureLU( LU_size, min_time, R);  
	break;
      }

     printf("tail %8.2f\n",lll);
   }


        res[0] = (res[1] + res[2] + res[3] + res[4] + res[5]) / 5;

        /* print out results  */
        printf("Composite Score:        %8.2f\n" ,res[0]);
        printf("FFT             Mflops: %8.2f    (N=%d)\n", res[1], FFT_size);
        printf("SOR             Mflops: %8.2f    (%d x %d)\n", 		
				res[2], SOR_size, SOR_size);
        printf("MonteCarlo:     Mflops: %8.2f\n", res[3]);
        printf("Sparse matmult  Mflops: %8.2f    (N=%d, nz=%d)\n", res[4], 
					Sparse_size_M, Sparse_size_nz);
        printf("LU              Mflops: %8.2f    (M=%d, N=%d)\n", res[5],
				LU_size, LU_size);

        printf("|  %8.2f|  %8.2f|  %8.2f|  %8.2f|  %8.2f|  %8.2f||\n", res[0],res[1],res[2],res[3],res[4],res[5]);

        Random_delete(R);

        return 0;
  
}

void print_banner()
{
 printf("**                                                              **\n");
 printf("** SciMark2 Numeric Benchmark, see http://math.nist.gov/scimark **\n");
 printf("** for details. (Results can be submitted to pozo@nist.gov)     **\n");
 printf("**                                                              **\n");
}
