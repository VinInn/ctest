void loop(double *b, double const * globalMatD, double const * x, int n, int indij, int i, int j, int lj) {
  double bb = b[i];
#pragma GCC ivdep
  for (int jj=1; jj<n; ++jj) {
    b[j]=b[j]+globalMatD[indij+lj]*x[i];
    bb=bb+globalMatD[indij+lj]*x[j];
    j=j+1;
    lj=lj+1;
  }
  b[i]=bb;
}
