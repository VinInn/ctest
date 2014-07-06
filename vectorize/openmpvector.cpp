int main()
{

 const unsigned int nEvents = 1000;
 double results[nEvents] = {0};
 double pData[nEvents] = {0};
 double coeff = 12.2;

#pragma omp parallel for
 for (int idx = 0; idx<(int)nEvents; idx++) {
   results[idx] = coeff*pData[idx];
 }

 return results[0]; // avoid optimization of "dead" code

}
