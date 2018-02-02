#ifndef __H_LINESEARCH__
#define __H_LINESEARCH__

#include <cuda_types.h>
#include <dataset.h>

real cg_linesearch (real *, real *, real , real , SparseDataset *, real *, real *, 
                        real , int , int , int, real *, real *, real *, real *, real * );

#endif
