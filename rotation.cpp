
#define RESTRICT __restrict__
// #define RESTRICT

void rotate(double a[3]) {
  double v[3] = { 1., 2., 3. };
  double r[9] = { 0.5, 0.3, 0.,
                  0.3, -0.5, 0.,
                  0., 0., 1.
  };

  for (int i=0; i<3; i++) {
//   a[i]=0; int j=3*i;
  // for(int k=0; k<3;k++) a[i] +=r[j+k]*v[k]; 

    int j=3*i;
    a[i] = r[j]*v[0] + r[j+1]*v[1] + r[j+2]*v[2];
  }

} 


void rotate(double const  __restrict__ v[3], double const  __restrict__ r[9], double  __restrict__ a[3]) {
  for (int i=0; i<3; i++) {
//   a[i]=0; int j=3*i;
  // for(int k=0; k<3;k++) a[i] +=r[j+k]*v[k]; 
    int j=3*i;
    a[i] = r[j]*v[0] + r[j+1]*v[1] + r[j+2]*v[2];
  }

} 

void rotate(double const __restrict__ v[3], float const __restrict__ r[9], double __restrict__ a[3]) {
  for (int i=0; i<3; i++) {
//   a[i]=0; int j=3*i;
  // for(int k=0; k<3;k++) a[i] +=r[j+k]*v[k]; 
    int j=3*i;
    a[i] = r[j]*v[0] + r[j+1]*v[1] + r[j+2]*v[2];
  }

} 
