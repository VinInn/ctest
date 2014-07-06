//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2012 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// ===============================================================
//
//   program to demonstrate run-to-run variations in floating-point results
//   due to variations in data alignment.  Compile with
//           icc run_to_run.cpp -vec-report2 -xavx
//   and run at 1 second intervals for 16 seconds;   then compile with
//           icc run_to_run.cpp -vec-report2 -xavx -fp-model precise
//   repeat and compare results. Note the reason given for why the
//   reduction loop at line 44 is not vectorized in the second case.
//
//       Martyn Corden
//
#include <cstdlib>
#include <sys/time.h>
#include <iostream>

float sumit( float * b, int nb) {
    float sum=0.;
    for (int i=0; i<nb; i++) sum = sum + b[i];
    return sum;
}

int main()   {
	float *a, *b, sum=0.f;
	int na, nb=57;
	struct timeval start;

	gettimeofday(&start, NULL);
	na = (int)start.tv_sec%16;
//     allocations; size of a depends on the time of day
//     allocating a string containing the time of day (or day of the week, or user name, ...)
//     could have the same effect
//     result depends on the size of a, even though a is never used
	a = (float *) malloc(4*na);
	b = (float *) malloc(4*nb);
        long *ib = (long *)&b[0];
        long alignb = (long)ib;
//     initialization
	for (int i=0; i<nb;  i++) {
            b[i]= (float)i * (2*(i%2)-1) * 0.315f;
        }
//     reduction
      sum = sumit(b,nb);
//    for (int i=0; i<nb; i++) sum = sum + b[i];

	std::cout.precision(7);
 	std::cout << "time in secs %16 = " << na  << "   start address = " << &b[0]
                  << "   alignment = " << alignb%64 << "   sum = " << sum << std::endl;
}
