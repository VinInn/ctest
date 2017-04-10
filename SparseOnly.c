#include <stdlib.h>

    /*
        held in compress-row format.  If the size of the matrix
        in MxN with nz nonzeros, then the val[] is the nz nonzeros,
        with its ith entry in column col[i].  The integer vector row[]
        is of size M+1 and row[i] points to the begining of the
        ith row in col[].  
    */

    void SparseCompRow_matmult( int M, double *y, double *val, int *row,
        int *col, double *x)
    {
        int r;
        int i;

        
            for (r=0; r<M; r++)
            {
                double sum = 0.0; 
                int rowR = row[r];
                int rowRp1 = row[r+1];
                for (i=rowR; i<rowRp1; i++)
                    sum += x[ col[i] ] * val[i];
                y[r] = sum;
            }
    }

    int  kernel_measureSparseMatMult(int N, int nz, int cycles)
    {
        /* initialize vector multipliers and storage for result */
        /* y = A*y;  */

        double *x = (double*) malloc(sizeof(double)*N);
        double *y = (double*) malloc(sizeof(double)*N);
        for (int i=0;i<N;++i) x[i]=0.1;

#if 0
        // initialize square sparse matrix
        //
        // for this test, we create a sparse matrix with M/nz nonzeros
        // per row, with spaced-out evenly between the begining of the
        // row to the main diagonal.  Thus, the resulting pattern looks
        // like
        //             +-----------------+
        //             +*                +
        //             +***              +
        //             +* * *            +
        //             +** *  *          +
        //             +**  *   *        +
        //             +* *   *   *      +
        //             +*  *   *    *    +
        //             +*   *    *    *  + 
        //             +-----------------+
        //
        // (as best reproducible with integer artihmetic)
        // Note that the first nr rows will have elements past
        // the diagonal.
#endif

        int nr = nz/N;      /* average number of nonzeros per row  */
        int anz = nr *N;    /* _actual_ number of nonzeros         */

            
        double *val = (double*) malloc(sizeof(double)*anz);
        for(int i=0;i<anz;++i) val[i]=0.1;
        int *col = (int*) malloc(sizeof(int)*nz);
        int *row = (int*) malloc(sizeof(int)*(N+1));
        int r=0;

        row[0] = 0; 
        for (r=0; r<N; r++)
        {
            /* initialize elements for row r */

            int rowr = row[r];
            int step = r/ nr;
            int i=0;

            row[r+1] = rowr + nr;
            if (step < 1) step = 1;   /* take at least unit steps */


            for (i=0; i<nr; i++)
                col[rowr+i] = i*step;
                
        }

        int res=0;
         for (int i=0; i<cycles;++i) {
            SparseCompRow_matmult(N, y, val, row, col, x);
            for (r=0; r<N; r++) y[0]+=y[r];
            res+= y[0]/10000000.;
         }

        free(row);
        free(col);
        free(val);
        free(y);
        free(x);

        return res;
    }


    int main() {
    /*
    //  const  int SPARSE_SIZE_M = 1000;
    //  const  int SPARSE_SIZE_nz = 5000;
   //   const  int LG_SPARSE_SIZE_M = 100000;
   //  const  int LG_SPARSE_SIZE_nz =1000000;
   */


     int N=1000; int nz=5000;
     int cycles=100000;
//     int N=100000; int nz=1000000;
//     int cycles=1000;
     
     return  kernel_measureSparseMatMult(N,nz,cycles);

    }
