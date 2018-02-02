
#ifndef __H_LOGISTIC_FN_INDICATOR__
#define __H_LOGISTIC_FN_INDICATOR__

#include "cuda_types.h"
#include "dataset.h"

void logistic_fn_indicator (real *features, SparseDataset *spfeatures, real *target, real *weights, real lambda, int rows, int cols, real *fx, real *devPtr, real *hostPtr);
void logistic_fn_indicator_gx (real *features, SparseDataset *spfeatures, real *target, real *weights, real lambda, int rows, int cols, real *gx, real *devPtr, real *hostPtr, int samplingType, int numFeatures);
void logistic_fn_indicator_hx_matvec (real *features, SparseDataset *spFeatures, real *weights, real *vector,
                                real lambda, int rows, int cols, real *hx, real *devPtr, real *hostPtr, int type, real *scale, int numFeatures);
void logistic_fn_indicator_hx (real *features, real *target, real *weights, real lambda, int rows, int cols, real *hx, real *devPtr, real *hostPtr);
void logistic_regression_predict( real *, SparseDataset *, real *, real *, real *, int , int , real *, real *, real *);

void logisticRegDerivativeTest ( real *features, real *target, int rows, int cols,
                        real *devPtr, real *hostPtr, real *pageLckPtr, int numpoints);


//Non uniform functions
int generateNonUniformSample_log( real *probs, real *scaleTerms, int rows, int sampleSize, int *selIndices, real *devPtr, real *hostPtr);
void computeRowProbabilities_log( SparseDataset *spfeatures, real *features, int rows, int cols,
                        real *dHXW, real *rowNrms, real *probs, real *devPtr );
void computeRowNorms_log( SparseDataset *spfeatures, real *features, int rows, int cols, real *rowNrms, real *devPtr );
void computeHXW_log (SparseDataset *spfeatures, real *features, int rows, int cols, real *weights, real *B );



#endif
