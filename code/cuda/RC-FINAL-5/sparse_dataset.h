#ifndef __H_SPARSE_DATASET__
#define __H_SPARSE_DATASET__

#include "dataset.h"

void convertToCSR( DeviceDataset *, real * ); 
void convertHessianSampleToCSR (SparseDataset *spSampleHessian, int sampleSize, int cols, real *devPtr);
void convertGradientSampleToCSR (SparseDataset *spSampleHessian, int sampleSize, int cols, real *devPtr);

void initMatDescriptors( DeviceDataset *d );
void initMatDescriptorsForSampling( DeviceDataset *d );
void initMatDescriptorsForSparseSampling( DeviceDataset *d );

#endif
