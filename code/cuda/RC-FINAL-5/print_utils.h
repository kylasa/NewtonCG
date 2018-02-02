#ifndef __H_PRINT_UTILS__
#define __H_PRINT_UTILS__

#include "cuda_types.h"

void printVector( real *src, int s, real *t );
void printCustomVector( real *src, int s, int jump );
void printIntVector( int *src, int s, int *t );
void printHostVector( real *src, int s  );
void writeMatrix (real *mat, int c);
void writeVector (real *mat, int c, char *file, int );
void writeIntVector (int *mat, int c );
void writeSparseMatrix (real *dataPtr, int *rowIndex, int *colIndex, int m, int n, int nnz );

real computeWeightSum( real *weights, int len ); 

int readVector( real *vec, int rows, char *file, int offset );



#endif
