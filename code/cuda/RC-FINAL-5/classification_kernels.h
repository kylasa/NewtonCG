#ifndef __H_CLASSIFICATION_KERNELS__
#define __H_CLASSIFICATION_KERNELS__

#include "cuda_types.h"

__device__ __inline__ double my_shfl(double x, int lane);
__device__ __inline__ double warpSum( double x );

GLOBAL void reduce(const real *input, real *results, const size_t count) ;
GLOBAL void ker_init_scaleTerms ( real *scaleTerms, int sampleSize, real *probs, int *indices );
GLOBAL void ker_compute_probs( real *probs, int rows, int sampleSize, real *randVec, real *indices);
GLOBAL void ker_compute_dHXW_nrm_log (real *dHXW, real *rowNrms, int rows);
GLOBAL void ker_normalize (real *dHXW, int rows, real *nrmConstant, real *probs );
GLOBAL void ker_row_norms( real *features, int rows, int cols, real *nrm );
GLOBAL void ker_sqr_elements ( real *ptr, int nnz, int elems_per_thread, real *results );
GLOBAL void ker_sqrt_elements (real *ptr, int count );
GLOBAL void ker_init_ones (real *ptr, int count );


#endif
