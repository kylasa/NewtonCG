#ifndef __H_NEWTON_CG__
#define __H_NEWTON_CG__

#include "cuda_types.h"
#include "dataset.h"

typedef struct cg_params{
	int max_iterations; 
	int max_cg_iterations;
	real tolerance; 
	real cg_tolerance;
	real iflag;
	real lambda;
	
	//Subsampling
	int gx_sampling; 
	int hx_sampling; 

} NEWTON_CG_PARAMS;

int newton_cg( ForestDataset *, DeviceDataset *, NEWTON_CG_PARAMS *, SCRATCH_AREA *);
int newton_cg_multi_optimized( ForestDataset *host, DeviceDataset *data, NEWTON_CG_PARAMS *params, SCRATCH_AREA *scratch );


#endif
