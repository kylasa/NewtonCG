#ifndef __SOFTMAX_MULTICLASS_H__
#define __SOFTMAX_MULTICLASS_H__

#include "cuda_types.h"
#include "dataset.h"


int generateNonUniformSample( real *probs, real *scaleTerm, int rows, int sampleSize, int *selIndices, real *devPtr, real *hostPtr);
void computeRowProbabilities( SparseDataset *spfeatures, real *features, int rows, int cols, int numclasses,
                        real *dHXW, real *rowNrms, real *probs, real *devPtr );
void computeRowNorms( SparseDataset *spfeatures, real *features, int rows, int cols, real *rowNrms, real *devPtr );
void computeDiagHXW( real *XW, int rows, int num_classes, real *dXW );


real softmax_multiclass_fx (SparseDataset *, real *, real *, int , int , int, real *,
                                real , real *, real *, real *);
void softmax_multiclass_gx (real *, real *, int , int ,
                                int , real *, real , real *, 
                                real *, real *, real *);
void softmax_multiclass_hx (real *, int , int , int ,
                                real *, real *, real ,
                                real *, real *, real *, real *, real *, int);

void computeHXW (SparseDataset *, real *features, int rows, int cols, int num_classes, real *weights, real *XW, int subSampling );

void computeExpSum( real *XW, int rows, int cols, int num_classes, real *expSumVec );

void softmax_multiclass_gx_optimized (SparseDataset *, real *features, real *target, int rows, int cols, int num_classes,
                        real *weights, real lambda, real *XW, real *gradient,
                        real *devPtr, real *hostPtr, real *pageLckPtr);

void softmax_multiclass_gx_subsampled(SparseDataset *, real *features, real *target, int rows, int cols, int num_classes,
                        real *weights, real lambda, real *gradient,
                        real *devPtr, real *hostPtr, real *pageLckPtr, 
			SparseDataset *, real *, SparseDataset *, real *, int, int );

void softmax_multiclass_hx_subsampled(SparseDataset *, real *features, int rows, int cols, int num_classes,
                                real *weights, real *vector, real lambda,
                                real *devPtr, real *hostPtr, real *pageLckPtr, real *Hv, real *B, 
				SparseDataset *, real *, SparseDataset *, int, real *, int );

void softmax_multiclass_hx_optimized (SparseDataset *, real *features, int rows, int cols, int num_classes,
                                real *weights, real *vector, real lambda,
                                real *devPtr, real *hostPtr, real *pageLckPtr, real *Hv, real *B ); 

real softmax_predict(SparseDataset *, real *, real *, real *, int , int , int ,
                        real *, real *, int, real *);

void expTest( real *results, int count, real *host);

void computeErrors ( real *, real *, int , int , int ,
                        real *, real *, real *, int );


void hostDerivativeTest ( real *, real *, int , int , int ,
                        real *, real *, int);

#endif
