#ifndef _H_CONJUGATE_GRADIENT__
#define _H_CONJUGATE_GRADIENT__

#include <cuda_types.h>
#include <dataset.h>
#include <newton_cg.h>


int Cublas_CG_Logistic( DeviceDataset *data, NEWTON_CG_PARAMS *params,
                real *g, real *x, real *x_best, real *rel_residual,
                real *devPtr, real *hostPtr, real *pgeLckPtr);
int Cublas_CG_multi_optimized (SparseDataset *,  real *, real *, real *, real *, real *, real, int , int , int ,
                        real *, real *, real *, real *, int , real, real *, real *, 
			SparseDataset *, real *, SparseDataset *, int, int );
#endif

