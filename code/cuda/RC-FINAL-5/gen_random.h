#ifndef __H_GEN_RANDOM__
#define __H_GEN_RANDOM__

#include "cuda_types.h"

void getRandomVector (int n, real *hostPtr, real *devPtr);

void randomShuffle( int *idx, int m ); 
void genRandomVector( int *idx, int m, int n );


#endif
