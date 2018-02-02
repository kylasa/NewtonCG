#ifndef __SUB_SAMPLING_HELPERS_H__
#define __SUB_SAMPLING_HELPERS_H__

#include "dataset.h"
#include "cuda_types.h"

void initSubSampledHessian( int offset, int rows, SparseDataset *spSampledHessian, real *, int sampledSize );
void prepareForNonUniformSampling (SparseDataset *samplingMat, int sampleSize, int *indices) ;


void prepareForSampling (SparseDataset *sampledHessian, real *, real *, int rows, int sampleSize, int *hostPtr);
void sampleDataset( SparseDataset *spSampledHessian, real *dataset,
                        int rows, int cols, int num_classes,
                        real *subSampledHessian, int sampleSize );

void sampleSparseDataset ( SparseDataset *spSampler, SparseDataset *spDataset,
                                int rows, int cols, int num_classes,
                                SparseDataset *spGradientSample, int sampleSize );

#endif
