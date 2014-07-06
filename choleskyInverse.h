// cholesky inversion...



template<typename M, size_t N>
bool choleskyInverse(M & m) {
  typedef typename M::value_type T;
  //  enum { N=typename M::kRows};
  T a[N][N];
  for (size_t i=0; i<N; ++i) {
    a[i][i]=m(i,i);
    for (size_t j=i+1; j<N; ++j) 
      // a[i][j] = 
	a[j][i] = m(i,j);
  }


  for (size_t j=0; j<N; ++j) {
    if(0>a[j][j]) return false;
    a[j][j]  =  1. / a[j][j];
    size_t jp1  =  j+1;
    for (size_t l=jp1; l<N; ++l) {
      a[j][l]  =  a[j][j]*a[l][j];
      T s1 =  -a[l][jp1];
      for (size_t i=0; i<jp1;++i)
	s1+= a[l][i]*a[i][jp1];
      a[l][jp1]  =  -s1;
    }
  }

  if(N==1)  return true;
  a[0][1]  =  -a[0][1];
  a[1][0]  =   a[0][1]*a[1][1];
  for (size_t j=2; j<N; ++j) {
    size_t jm1 = j - 1;
    for (size_t k=0; k<jm1; ++k) {
      T s31  =  a[k][j];
      for (size_t i=k; i<jm1; ++i)
	s31  += a[k][i+1]*a[i+1][j];
      a[k][j]  =  -s31;
      a[j][k]  =  -s31*a[j][j];
    }
    a[jm1][j]  =  -a[jm1][j];
    a[j][jm1]  =   a[jm1][j]*a[j][j];
  }
  
  size_t j=0;
  while (j<N-1) { 
    T s33  =  a[j][j];
    for (size_t i=j+1; i<N; ++i) 
      s33  +=  a[j][i]*a[i][j];
    //    a[j][j]  =  s33;
    m(j,j) = s33;

    ++j;
    // if (j==N) break;
    for (size_t k = 0; k<j; ++k) {
      T s32  = 0;
      for (size_t i=j; i<N; ++i)
	s32  +=  a[k][i]*a[i][j];
      //      a[j][k]= a[k][j]  =  s32;
      m(k,j) = s32;
    }
  }
  m(j,j)=a[j][j];
  
  /*
  for (size_t i=0; i<N; ++i) {
    m(i,i) = a[i][i];
    for (size_t j=i+1; j<N; ++j) 
      m(i,j) = a[i][j];
  }
  */
  return true;
}
