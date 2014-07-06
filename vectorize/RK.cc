
double H0[4], H1[4], H2[4];
void RKS(double * __restrict__ dR, double * __restrict__ dA, double S3) {

  for(int i=0; i<35; i+=7) {
    //double* __restrict__ dR = &P[i];
    // double* __restrict__ dA = &P[i+3];
    double dA0 = H0[ 2]*dA[1]-H0[ 1]*dA[2];
    double dB0 = H0[ 0]*dA[2]-H0[ 2]*dA[0];
    double dC0 = H0[ 1]*dA[0]-H0[ 0]*dA[1];
    //   if(i==35) {dA0+=A0; dB0+=B0; dC0+=C0;}
    double dA2 = dA0+dA[0];
    double dB2 = dB0+dA[1];
    double dC2 = dC0+dA[2];
    double dA3 = dA[0]+dB2*H1[2]-dC2*H1[1];
    double dB3 = dA[1]+dC2*H1[0]-dA2*H1[2];
    double dC3 = dA[2]+dA2*H1[1]-dB2*H1[0];
    //if(i==35) {dA3+=A3-A00; dB3+=B3-A11; dC3+=C3-A22;}
    double dA4 = dA[0]+dB3*H1[2]-dC3*H1[1];
    double dB4 = dA[1]+dC3*H1[0]-dA3*H1[2];
    double dC4 = dA[2]+dA3*H1[1]-dB3*H1[0];
    // if(i==35) {dA4+=A4-A00; dB4+=B4-A11; dC4+=C4-A22;}
    double dA5 = dA4+dA4-dA[0];
    double dB5 = dB4+dB4-dA[1];
    double dC5 = dC4+dC4-dA[2];
    double dA6 = dB5*H2[2]-dC5*H2[1];
    double dB6 = dC5*H2[0]-dA5*H2[2];
    double dC6 = dA5*H2[1]-dB5*H2[0];
    // if(i==35) {dA6+=A6; dB6+=B6; dC6+=C6;}
    dR[0]+=(dA2+dA3+dA4)*S3; dA[0]=(dA0+dA3+dA3+dA5+dA6)*.33333333;
    dR[1]+=(dB2+dB3+dB4)*S3; dA[1]=(dB0+dB3+dB3+dB5+dB6)*.33333333;
    dR[2]+=(dC2+dC3+dC4)*S3; dA[2]=(dC0+dC3+dC3+dC5+dC6)*.33333333;
    dR+=7;
    dA+=7;
  }
  
}
