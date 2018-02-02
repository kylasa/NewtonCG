#ifndef _H_CUDA_ENVIRONMENT__
#define _H_CUDA_ENVIRONMENT__

#include "cuda_types.h"

void cuda_env_init (SCRATCH_AREA *, int);
void cuda_env_cleanup (SCRATCH_AREA *);

#endif
