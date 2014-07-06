void nestedCond( double * __restrict__ x_in,  double * __restrict__ x_out,  double * __restrict__ a,  double * __restrict__ c, int M, int N) {   
for (int j = 0; j < 256; j++)
    {
      double x = x_in[j];
      double curr_a = a[0];

      for (int i = 0; i < 256; i++)
        {
          double next_a = a[i+1];
          curr_a = x > c[i] ? curr_a : next_a;
        }

      x_out[j] = curr_a;
    }
}
