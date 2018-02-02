#include "mat_functions.h"
#include "cuda.h"
#include "cuda_runtime.h"

GLOBAL void ker_log_sum( real *t, real *target, int N, real *out)
{
	//extern __shared__ real sdata[];
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	real x = 0; 

	if (idx < N) {	
		x = t[ idx ]; 	
		if (x <= 0)
			out[ idx ] = log( 1. + exp(x) ) - ((target[idx] - 1.) * t[ idx ]); 
		else 
			out[ idx ] = ( x + log( exp(-x) + 1.) ) - ((target[idx] - 1.) * t[ idx] );
	}
}

GLOBAL void ker_sigmoid( real *s, int N, real *out)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	real x = 0; 
	real alpha = 0; 

	if (idx < N) {
		x = s[ idx ];
		if ( x < 0 ) 
			out[ idx ] = exp( x ) / (1. + exp(x) ); 
		else
			out[ idx ] = 1. / (1. + exp(-x) ); 
	}
}

GLOBAL void ker_sigmoid_classify( real *s, int N )
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (idx < N) {
		if (s[ idx ] <= 0 ){ 
			if (exp(s[idx])/ ( (1. + exp(s[idx]) )) > 0.5)
				s[idx] = 1.;
			else 
				s[idx] = 0.;
		} else {
			if (1. / (1. + exp(-s[idx]) ) > 0.5)
				s[idx] = 1.;
			else 
				s[idx] = 0.;
		}
	}
}

GLOBAL void ker_sigmoid_target( real *t, real *target, int N, real *out)
{
	real x = 0; 
	real alpha = 0; 
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (idx < N) {
		x = t[ idx ];
		if (x < 0 ) 
			out[idx] = ( exp(x)/ ( 1. + exp(x) )) - (target[ idx ] - 1.);
		else
			out[idx] = ( 1./ ( 1. + exp(-x) )) - (target[ idx ] - 1.);
	}
}

GLOBAL void ker_ele_vec_product( real *t1, real *t2, int N, real *out)
{
	//extern __shared__ real sdata[];
	//real x = 0; 
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (idx < N) out[ idx ] = t1[ idx ] * t2[ idx ];
	//sdata[ threadIdx.x ] = x; 
	//if (idx < N) out[idx] = sdata[threadIdx.x] ;
}

GLOBAL void ker_mat_identity( real *matrix, real gamma, int M)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < M)
		matrix[ idx * M + idx ] += gamma;
}

GLOBAL void ker_hx_matvec_reg ( real *hx, real gamma, real *vec, int c)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < c) {
		hx[ idx ]+= gamma * vec[ idx ];
	}
}


GLOBAL void ker_reduction(const real *input, real *per_block_results, int n)
{
  extern __shared__ real sdata[];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  real x = 0;

  if(i < n)
  {
    x = input[i];
  }
  sdata[threadIdx.x] = x;
  __syncthreads();

  for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
    {   
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
    }   

    __syncthreads();
  }

  if(threadIdx.x == 0)
  {
    per_block_results[blockIdx.x] = sdata[0];
  }
}

