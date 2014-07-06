typedef struct
{
  int m[17];                        
  int seed;                             
  int i;                                /* originally = 4 */
  int j;                                /* originally =  16 */
  int /*boolean*/ haveRange;            /* = false; */
  double left;                          /*= 0.0; */
  double right;                         /* = 1.0; */
  double width;                         /* = 1.0; */
}
Random_struct, *Random;

Random new_Random_seed(int seed);
double Random_nextDouble(Random R);
void Random_delete(Random R);
double *RandomVector(int N, Random R);
double **RandomMatrix(int M, int N, Random R);
#include <stdlib.h>


#ifndef NULL
#define NULL 0
#endif


  /* static const int mdig = 32; */
#define MDIG 32

  /* static const int one = 1; */
#define ONE 1

  static const int m1 = (ONE << (MDIG-2)) + ((ONE << (MDIG-2) )-ONE);
  static const int m2 = ONE << MDIG/2;

  /* For mdig = 32 : m1 =          2147483647, m2 =      65536
     For mdig = 64 : m1 = 9223372036854775807, m2 = 4294967296 
  */

                                /* move to initialize() because  */
                                /* compiler could not resolve as */
                                /*   a constant.                 */

  static /*const*/ double dm1;  /*  = 1.0 / (double) m1; */


/* private methods (defined below, but not in Random.h */

static void initialize(Random R, int seed);

Random new_Random_seed(int seed)
{
    Random R = (Random) malloc(sizeof(Random_struct));

    initialize(R, seed);
    R->left = 0.0;
    R->right = 1.0;
    R->width = 1.0;
    R->haveRange = 0 /*false*/;

    return R;
}

Random new_Random(int seed, double left, double right) 
{
    Random R = (Random) malloc(sizeof(Random_struct));

    initialize(R, seed);
    R->left = left;
    R->right = right;
    R->width = right - left;
    R->haveRange = 1;          /* true */

    return R;
}

void Random_delete(Random R)
{
    free(R);
}



/* Returns the next random number in the sequence.  */

double Random_nextDouble(Random R) 
{
    int k;

    int I = R->i;
    int J = R->j;
    int *m = R->m;

    k = m[I] - m[J];
    if (k < 0) k += m1;
    R->m[J] = k;

    if (I == 0) 
        I = 17;
    I--;
    R->i = I;

    if (J == 0) 
        J = 17 ;
    J--;
    R->j = J;

    if (0==R->haveRange) 
       return dm1 * (double) k;
     return  R->left +  dm1 * (double) k * R->width;
/*    else */
/*    return dm1 * (double) k;*/

} 




/*--------------------------------------------------------------------
                           PRIVATE METHODS
  ----------------------------------------------------------------- */

static void initialize(Random R, int seed) 
{

    int jseed, k0, k1, j0, j1, iloop;

    dm1  = 1.0 / (double) m1; 

    R->seed = seed;

    if (seed < 0 ) seed = -seed;            /* seed = abs(seed) */  
    jseed = (seed < m1 ? seed : m1);        /* jseed = min(seed, m1) */
    if (jseed % 2 == 0) --jseed;
    k0 = 9069 % m2;
    k1 = 9069 / m2;
    j0 = jseed % m2;
    j1 = jseed / m2;
    for (iloop = 0; iloop < 17; ++iloop) 
    {
        jseed = j0 * k0;
        j1 = (jseed / m2 + j0 * k1 + j1 * k0) % (m2 / 2);
        j0 = jseed % m2;
        R->m[iloop] = j0 + m2 * j1;
    }
    R->i = 4;
    R->j = 16;

}

double *RandomVector(int N, Random R)
{
    int i;
    double *x = (double *) malloc(sizeof(double)*N);

    for (i=0; i<N; i++)
        x[i] = Random_nextDouble(R);

    return x;
}


double **RandomMatrix(int M, int N, Random R)
{
    int i;
    int j;

    /* allocate matrix */

    double **A = (double **) malloc(sizeof(double*)*M);

    if (A == NULL) return NULL;

    for (i=0; i<M; i++)
    {
        A[i] = (double *) malloc(sizeof(double)*N);
        if (A[i] == NULL) 
        {
            free(A);
            return NULL;
        }
        for (j=0; j<N; j++)
            A[i][j] = Random_nextDouble(R);
    }
    return A;
}




/**
 Estimate Pi by approximating the area of a circle.

 How: generate N random numbers in the unit square, (0,0) to (1,1)
 and see how are within a radius of 1 or less, i.e.
 <pre>  

 sqrt(x^2 + y^2) < r

 </pre>
  since the radius is 1.0, we can square both sides
  and avoid a sqrt() computation:
  <pre>

    x^2 + y^2 <= 1.0

  </pre>
  this area under the curve is (Pi * r^2)/ 4.0,
  and the area of the unit of square is 1.0,
  so Pi can be approximated by 
  <pre>
                # points with x^2+y^2 < 1
     Pi =~      --------------------------  * 4.0
                     total # points

  </pre>

*/

static const int SEED = 113;


    double MonteCarlo_num_flops(int Num_samples)
    {
        /* 3 flops in x^2+y^2 and 1 flop in random routine */

        return ((double) Num_samples)* 4.0;

    }

    

    double MonteCarlo_integrate(int Num_samples)
    {


        Random R = new_Random_seed(SEED);


        int under_curve = 0;
        int count;

        for (count=0; count<Num_samples; count++)
        {
            double x= Random_nextDouble(R);
            double y= Random_nextDouble(R);

            if ( x*x + y*y <= 1.0)
                 under_curve ++;
            
        }

        Random_delete(R);

        return ((double) under_curve / Num_samples) * 4.0;
    }


    int main()
    {
        double result = 0.0;

        int cycles=100000000;
        MonteCarlo_integrate(cycles);

        return 0;
    }


