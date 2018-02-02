#include "softmax_multiclass.h"
#include "cuda_utils.h"

#include "gen_random.h"
#include "cuda_types.h"
#include "print_utils.h"

#include "classification_kernels.h"

GLOBAL void ker_exp( real *results, int count)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	if (idx < count)
		results[idx] = exp( (real)idx );
}

void expTest( real *results, int count, real *host){

	ker_exp <<< 1, count>>> (results, count);
	cudaThreadSynchronize ();
	cudaCheckError ();
}

__device__ __inline__ double my_shfl(double x, int lane)
{
        // Split the double number into 2 32b registers.
        int lo, hi;
        asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x));

        // Shuffle the two 32b registers.
        lo = __shfl_xor(lo, lane);
        hi = __shfl_xor(hi, lane);

        // Recreate the 64b number.
        //asm volatile( "mov.b64 %0, {%1,%2};" : "=d(x)" : "r"(lo), "r"(hi));
        //return x;
        return __hiloint2double( hi, lo);
}

__device__ __inline__ double warpSum( double x )
{
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
                x += my_shfl( x, offset);
        return x;
}


GLOBAL void ker_add_regularizer ( real *input, real *vector, real lambda, int count, real normalizer)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (idx < count) input[ idx ] += lambda * vector[ idx ] ;
}


/*
GLOBAL void reduce(const real *input, real *results, const size_t count) {
        extern __shared__ real my_results[];
        unsigned int lane = threadIdx.x >> 5;
        unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

        real sdata;
        real x = 0;

        sdata = 0;
        my_results[ lane ] = 0;
        if(idx < count) x = input [idx];
        sdata = x;

        sdata = warpSum ( sdata );
        if (threadIdx.x % WARP_SIZE == 0) my_results[lane] = sdata;
        __syncthreads ();

        if (blockDim.x/WARP_SIZE == 0)
        	sdata = (threadIdx.x < 1) ? my_results[threadIdx.x] : 0;
        else
        	sdata = (threadIdx.x < (blockDim.x/WARP_SIZE)) ? my_results[threadIdx.x] : 0;
        __syncthreads ();

        if (lane == 0) sdata = warpSum( sdata );
        if(threadIdx.x == 0) results [ blockIdx.x  ] =  sdata;
}

*/

GLOBAL void reduce_vector_warp( const real *input, const real *maxdots, real *results, const size_t numcomps, int numblocks )
{
	extern __shared__ real my_results[]; 

	unsigned int lane  = threadIdx.x >> 5; 
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x; 

	real sdata; 
        sdata = 0.;

	if (idx < numcomps ){
		for (int c = 0; c < numblocks; c ++) sdata += input [ c * numcomps + idx ]; 
		results[ idx ] = sdata + exp( -1. * maxdots[ idx ] ); 
	}
}


GLOBAL void reduce_vector_warp_mt( const real *input, const real *maxdots, real *results, const size_t numcomps, int numblocks )
{
	unsigned int col =  threadIdx.x >> 5; 
        unsigned int myRowId = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE; 

	real sdata; 
	real x = 0.;

        sdata = 0.;
	x = 0.0;
	if ((col < numblocks) && (myRowId < numcomps)) x = input[(col * numcomps) + myRowId ];
        sdata = x;
	__syncthreads ();

        sdata = warpSum ( sdata );
        if ((col == 0) && (myRowId < numcomps))	
		results [ myRowId ] = sdata + exp( -1 * maxdots[myRowId] );
}


GLOBAL void reduce_vector_mt( const real *input, real *results, const size_t numcomps, const real normalizer, int numblocks )
{
	extern __shared__ real my_results[]; 

	unsigned int idx =  threadIdx.x; 
        unsigned int lane = threadIdx.x >> 5;
	unsigned int compOffset = blockIdx.x; 

	real sdata; 
	real x = 0.;

	for (int i = compOffset; i < numcomps; i += gridDim.x){

        	sdata = 0.;
        	my_results[ lane ] = 0.;
		x = 0.0;
		if ((idx < numblocks) && (i < numcomps)) x = input[(idx * numcomps) + i ];
        	sdata = x;
		__syncthreads ();

        	sdata = warpSum ( sdata );
        	if (threadIdx.x % WARP_SIZE == 0) my_results[lane] = sdata;
        	__syncthreads ();

        	if (blockDim.x/WARP_SIZE == 0)
        		sdata = (threadIdx.x < 1) ? my_results[threadIdx.x] : 0;
        	else
        		sdata = (threadIdx.x < (blockDim.x/WARP_SIZE)) ? my_results[threadIdx.x] : 0;
        	__syncthreads ();

        	if (lane == 0) sdata = warpSum( sdata );
        	if((threadIdx.x == 0) && (i < numcomps))  results [ i ] =  sdata * normalizer;
        	__syncthreads ();
	}
}

GLOBAL void reduce_vector(const real *input, real *results, const size_t numclasses, const size_t cols, const real normalizer, int numblocks)
{
        extern __shared__ real my_results[];
        unsigned int lane = threadIdx.x >> 5;
        unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

        real sdata;
        real x = 0.;

	for (int i = 0; i < numclasses * cols; i ++){
        	sdata = 0.;
        	my_results[ lane ] = 0.;
		x = 0.0;
		if (idx < numblocks) x = input[idx * numclasses * cols + i];
        	sdata = x;
		__syncthreads ();

        	sdata = warpSum ( sdata );
        	if (threadIdx.x % WARP_SIZE == 0) my_results[lane] = sdata;
        	__syncthreads ();

        	if (blockDim.x/WARP_SIZE == 0)
        		sdata = (threadIdx.x < 1) ? my_results[threadIdx.x] : 0;
        	else
        		sdata = (threadIdx.x < (blockDim.x/WARP_SIZE)) ? my_results[threadIdx.x] : 0;
        	__syncthreads ();

        	if (lane == 0) sdata = warpSum( sdata );
        	if(threadIdx.x == 0) results [ i ] =  sdata * normalizer;
        	__syncthreads ();
	}
}

GLOBAL void reduce_log(const real *input, real *results, const size_t count) {
        extern __shared__ real my_results[];
        unsigned int lane = threadIdx.x >> 5;
        unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

        real sdata;
        real x = 0;

        sdata = 0;
        my_results[ lane ] = 0;
        if(idx < count) x = log(input [idx] );
        sdata = x;

        sdata = warpSum ( sdata );
        if (threadIdx.x % WARP_SIZE == 0) my_results[lane] = sdata;
        __syncthreads ();

        if (blockDim.x/WARP_SIZE == 0)
        	sdata = (threadIdx.x < 1) ? my_results[threadIdx.x] : 0;
        else
        	sdata = (threadIdx.x < (blockDim.x/WARP_SIZE)) ? my_results[threadIdx.x] : 0;
        __syncthreads ();

        if (lane == 0) sdata = warpSum( sdata );
        if(threadIdx.x == 0) results [ blockIdx.x  ] =  sdata;
}

GLOBAL void ker_compute_expsum( real *XW, int rows, int cols, int numclasses, 
			real *expSumVec, int threads_per_col)
{
	int myColId = ( blockIdx.x * blockDim.x + threadIdx.x ) % threads_per_col; 	
	int myRowId = ( blockIdx.x * blockDim.x + threadIdx.x ) / threads_per_col; 	
	
	//local Data. 
	real sdata = 0; 
	
	for (int i = myRowId; i < rows; i += gridDim.x * blockDim.x )
	{
		sdata = 0; 
	
		for (int j = myColId; j < cols; j ++ ) sdata += exp ( XW[ j * rows + i ] ); 

		//warp sum here. 
        	for (int offset = threads_per_col/2; offset > 0; offset /= 2) 
			sdata += my_shfl( sdata, offset);

		if (myColId == 0) expSumVec[ i ] = sdata; 
	}
}

/*

GLOBAL void ker_init_scaleTerms ( real *scaleTerms, int sampleSize, real *probs, int *indices )
{
        int myRowId = blockIdx.x * blockDim.x + threadIdx.x;
        if (myRowId < sampleSize){
                scaleTerms[ myRowId ] = probs[ indices[ myRowId ] ] ;
        }
}


GLOBAL void ker_compute_probs( real *probs, int rows, int sampleSize, real *randVec, real *indices)
{
        int myRowId = blockIdx.x * blockDim.x + threadIdx.x;
        if (myRowId < rows ){
                probs[ myRowId ] *= sampleSize;
                if (probs[ myRowId ] > 1.0) probs[ myRowId ] = 1.0;

                if (randVec[ myRowId ] < probs[ myRowId ] )
                        indices[ myRowId ] = 1;
                else
                        indices[ myRowId ] = 0;
        }
}

*/

GLOBAL void ker_compute_dHXW_nrm (real *dHXW, real *rowNrms, int rows, int numclasses)
{
        int myRowId = blockIdx.x * blockDim.x + threadIdx.x;

        if (myRowId < rows)
        {
                for (int j = 0; j < numclasses; j += 1 ){
                        dHXW[ j * rows + myRowId ] = abs( dHXW[ j * rows + myRowId ] * (1. - dHXW[ j * rows + myRowId ]) ) * rowNrms[ myRowId ];
                }
                for (int j = 1; j < numclasses; j += 1 ){
                        dHXW[ myRowId ] += dHXW[ j * rows + myRowId ];
                }
        }
}

/*

GLOBAL void ker_normalize (real *dHXW, int rows, real *nrmConstant, real *probs ){
        int myRowId = blockIdx.x * blockDim.x + threadIdx.x;
        if (myRowId < rows){
                probs[ myRowId ] = dHXW[ myRowId ] / nrmConstant[0];
        }
}

GLOBAL void ker_row_norms( real *features, int rows, int numclasses, real *nrm )
{
        int myRowId = ( blockIdx.x * blockDim.x + threadIdx.x );
        int i = 0;
        real sum = 0;

        if (myRowId < rows) {
                i = myRowId;
                for (int j = 0; j < numclasses; j += 1)
                        sum += pow( features[ j * rows + i ], 2.);

                nrm[ i ] = sqrt( sum );
        }
}


GLOBAL void ker_sqr_elements ( real *ptr, int nnz, int elems_per_thread, real *results )
{
        int myID = blockIdx.x * blockDim.x + threadIdx.x ;
        int i = 0;

        if (myID < nnz) {
                i = myID;
                //results[ i ] = ptr[ i ] * ptr[ i ];
                ptr[ i ] *= ptr[ i ];
        }

}

GLOBAL void ker_sqrt_elements (real *ptr, int count )
{
        int myRowId = ( blockIdx.x * blockDim.x + threadIdx.x );
        int i = 0;

        if (myRowId < count ){
                i = myRowId;
                ptr[ i ] = sqrt( ptr[ i ] );
        }
}

GLOBAL void ker_init_ones (real *ptr, int count )
{       
        int myRowId = ( blockIdx.x * blockDim.x + threadIdx.x );
        int i = 0;
        
        if (myRowId < count ){
                i = myRowId;
                ptr[ i ] = 1.0;
        }
}

*/


GLOBAL void ker_compute_HXW( real *XW, int rows, int cols, int numclasses, int threads_per_col )
{
	int myColId = ( blockIdx.x * blockDim.x + threadIdx.x ) % threads_per_col; 	
	int myRowId = ( blockIdx.x * blockDim.x + threadIdx.x ) / threads_per_col; 	
	int myWarpId = (blockIdx.x * blockDim.x + threadIdx.x ) % WARP_SIZE; 

	real sdata = 0; 
	int i = 0; 

	real maxdot = 0; 

	//for (int i = myRowId; i < rows; i += gridDim.x * blockDim.x){
	if (myRowId < rows) {
		i = myRowId;

		maxdot = 0; 
		for (int j = 0; j < numclasses; j += threads_per_col ) {
			if (maxdot < XW[ j * rows + i ]) maxdot = XW[ j * rows + i ]; 
		}

		sdata = 0; 
		for (int j = 0; j < numclasses; j += threads_per_col ) sdata += exp ( XW[ j * rows + i ] - maxdot ); 

		//for (int offset = threads_per_col/2; offset > 0; offset /= 2) sdata += my_shfl( sdata, myWarpId + offset ); 

		for (int j = 0; j < numclasses; j += threads_per_col ) 
			XW[ j * rows + i ] = exp( XW[ j * rows + i ] - maxdot ) / (exp(-1. * maxdot) + sdata); 
	}
}
		

GLOBAL void ker_compute_fx (real *matvec, int rows, int cols, int numclasses, 
				real *target, real *indicatorVal, int NUM_THREADS, real *maxdots )
{
	extern __shared__ real my_results[];

	int idx =  blockIdx.x * blockDim.x + threadIdx.x; 
	int myClrId = idx % NUM_THREADS; 
	int myRowId = idx / NUM_THREADS; 
        unsigned int lane = threadIdx.x >> 5;

	real sdata = 0; 
	real maxdot = 0; 
	
	//if (myRowId < rows) {
	for (int r = myRowId; r < rows; r += gridDim.x * blockDim.x ) {
		maxdot = 0; 
		 for (int i = myClrId; i < numclasses; i += NUM_THREADS){
			if (maxdot < matvec[ i * rows + r ]) maxdot = matvec[ i * rows + r]; 
	 	 }

		maxdots[ r ] = maxdot; 

		 for (int i = myClrId; i < numclasses; i += NUM_THREADS){
			if ((int)target[ r ] == (i + 1)) sdata += matvec[ i * rows + r ]; 
			matvec[ i * rows + r ] = exp( matvec[ i * rows + r ]  - maxdot); 
		 } 
	}
	__syncthreads (); 

        sdata = warpSum ( sdata );
        if (threadIdx.x % WARP_SIZE == 0) my_results[lane] = sdata ;
        __syncthreads ();

        if (blockDim.x/WARP_SIZE == 0)
        	sdata = (threadIdx.x < 1) ? my_results[threadIdx.x] : 0;
        else
        	sdata = (threadIdx.x < (blockDim.x/WARP_SIZE)) ? my_results[threadIdx.x] : 0;
        __syncthreads ();

        if (lane == 0) sdata = warpSum( sdata );
        if(threadIdx.x == 0) indicatorVal [ blockIdx.x  ] =  sdata;
}

GLOBAL void ker_softmax (real *features, real *target, int rows, int cols, int num_classes, 
			real *weights, real lambda, real *wspace )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int lane = threadIdx.x >> 5; 

	extern __shared__ real sh_vec[];
	real dot = 0;
	int myclass = 0; 

	real blk_sum = 0; 
	real psum = 0;

	for (int i = 0; i < num_classes; i ++){
		if (threadIdx.x < cols) sh_vec[threadIdx.x] = weights[i * cols + threadIdx.x];	
		__syncthreads ();
	
		dot = 0;
		if (idx  < rows ) {
			for (int j = 0; j < cols; j ++) dot += sh_vec[j] * features[ j  * rows + idx ]; 	
			psum += exp (dot);
		}
		__syncthreads ();
	}
	
	// subtract the weights * feature for the class it belongs. 
	if (idx < rows){ 
		psum =  log( 1 + psum );
		myclass = (int)(target[ idx ] - 1);
	}
	
	if ( idx < rows && myclass < num_classes) {
		dot = 0;
		for (int j = 0; j < cols; j ++) 
			dot += features[ j * rows + idx ] * weights[ myclass * cols + j ];
		psum = psum - dot;
	}
	__syncthreads ();

	// block reduction here. 
        blk_sum  = warpSum( psum );
        if (threadIdx.x % WARP_SIZE == 0) sh_vec[lane] = blk_sum;
        __syncthreads ();

        if (blockDim.x/WARP_SIZE == 0)
        	blk_sum = (threadIdx.x < 1) ? sh_vec[threadIdx.x] : 0;
        else
        	blk_sum = (threadIdx.x < (blockDim.x / WARP_SIZE) ) ? sh_vec[ threadIdx.x ] : 0;
        __syncthreads ();

        if (lane == 0) blk_sum = warpSum( blk_sum );
        if (threadIdx.x == 0) wspace[ blockIdx.x ] = blk_sum;
}

GLOBAL void ker_dx_softmax (real *features, real *target, int rows, int cols, int num_classes, 
			real *weights, real lambda, real *wspace )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int lane = threadIdx.x >> 5;
	extern __shared__ real sh_vec[];

	real numerator = 0.; 
	real denominator = 0.; 
	int indicator = 0; 
	real multiplier = 0.; 
	real blk_sum = 0.; 
	real p_i = 0.; 

	real maxdot = 0.;

	if (idx < rows) indicator = (int)(target[ idx ] - 1.); 
	__syncthreads ();

        //maxdot here. 
        for (int i = 0; i < num_classes; i ++){
                if (threadIdx.x < cols) sh_vec[threadIdx.x] = weights[i * cols + threadIdx.x ];
                __syncthreads ();

                numerator = 0.;
                if (idx < rows) {
                        for (int j = 0; j < cols; j ++)
                                numerator += sh_vec[j] * features[ j * rows + idx ];

                        if (maxdot < numerator) maxdot = numerator;
                }
                __syncthreads ();
        }


	//denominator here. 
	for (int i = 0; i < num_classes; i ++){
		if (threadIdx.x < cols) sh_vec[threadIdx.x] = weights[i * cols + threadIdx.x ];	
		__syncthreads ();
	
		numerator = 0.;
		if (idx < rows) {
			for (int j = 0; j < cols; j ++)
				numerator += sh_vec[j] * features[ j * rows + idx ]; 	
			denominator  += exp( numerator - maxdot );
		}
		__syncthreads ();
	}

	//numerator here. 
	//dw_i (j) here. 
	for (int i = 0; i < num_classes; i ++){
		if (threadIdx.x < cols) sh_vec[threadIdx.x] = weights[i * cols + threadIdx.x];	
		__syncthreads ();

		numerator = 0; 
		if ( idx < rows ){
			for (int j = 0; j < cols; j ++)
				numerator += sh_vec[j] * features[ j * rows + idx ]; 	
			numerator = exp( numerator - maxdot );
			//p_i = numerator / (1 + denominator); 
			p_i = numerator / (exp(1. * maxdot) + denominator); 

			if (i == indicator) multiplier = 1.0;
			else multiplier = 0.;
		}
		__syncthreads ();

		for (int j = 0; j < cols; j ++){ 
			blk_sum = 0.; 
			if (idx < rows)
				blk_sum = (p_i - multiplier) * features[ j * rows + idx ];
			
        		__syncthreads ();

			// block level reduction here. 
        		blk_sum  = warpSum( blk_sum);
        		if (threadIdx.x % WARP_SIZE == 0) sh_vec[lane] = blk_sum;
        		__syncthreads ();

        		if (blockDim.x/WARP_SIZE == 0)
        			blk_sum = (threadIdx.x < 1) ? sh_vec[threadIdx.x] : 0;
        		else
        			blk_sum = (threadIdx.x < (blockDim.x / WARP_SIZE) ) ? sh_vec[ threadIdx.x ] : 0;
        		__syncthreads ();

        		if (lane == 0) blk_sum = warpSum( blk_sum );
        		if (threadIdx.x == 0) wspace[ (blockIdx.x * num_classes * cols) +  ( i * cols + j )  ] = blk_sum;
        		__syncthreads ();
		}
	}
}

GLOBAL void ker_dx_softmax_mt (real *features, real *target, int rows, int cols, int num_classes, 
			real *weights, real lambda, real *XW, real *expSum, real *wspace, int threads_per_col)
{
	extern __shared__ real shmem[];

	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	int myIdx = idx / threads_per_col; 

	real indicator = 0; 
	real class_prob;

	for (int clr = 0; clr < num_classes; clr ++){

		for (int col = myIdx; col < cols; col += threads_per_col){

			shmem[ myIdx ] = 0; 

			for (int r = 0; r < rows; r += gridDim.x * blockDim.x ) {
				class_prob = XW[ clr * rows + r ] / expSum[ r ];
				if (clr == (target[ r ] - 1)) indicator = 1. ; 
				shmem[ myIdx ] += (class_prob - indicator) * features[ col * rows + r ];
			}

			wspace[ blockIdx.x * num_classes * cols + clr * cols + col ] = shmem[ myIdx ]; 
		}
	}
}

GLOBAL void ker_dx_softmax_ind( real *hxw, real *target, int rows, int num_classes, real *result, int threads_per_row)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	int myClrId = idx % threads_per_row; 	
	int myRowId = idx / threads_per_row; 

	real indicator = 0; 
	int r = 0; 
	
	//for (int r = idx; r < rows; r += gridDim.x * blockDim.x){
	
	if (idx < rows ) {
		r = idx; 
		for (int clr = 0; clr < num_classes; clr ++ ){
			result[ clr * rows + r ] = hxw[ clr * rows + r ];
			if (clr == (int)(target[ r ] - 1.)) result[ clr * rows + r ] -= 1.; 

			//result[ clr * rows + r ] = 0;
			//if (clr == (int)(target[ r ] - 1.)) result[ clr * rows + r ] = 1; 
		}
	}
}

////Hessian functions here. 
GLOBAL void ker_hx_Xv ( real *features, real *vector, int rows, int cols, int num_classes, real *A ) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	extern __shared__ real sh_vec[];

	real dot = 0; 

	for (int i = 0; i < num_classes; i ++){
		if (threadIdx.x < cols) sh_vec[threadIdx.x] = vector [i * cols + threadIdx.x ];	
		__syncthreads ();
	
		if (idx < rows) {
			dot = 0; 
			for (int j = 0; j < cols; j ++) dot += sh_vec[j] * features[ j * rows + idx ]; 	
			A[ idx + i * rows ] = dot;  // column major format here. 
		}
		__syncthreads ();
	}
}

GLOBAL void ker_hx_ProbabilityTerms ( real *features, real *weights, int rows, int cols, int num_classes, real *B ) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	extern __shared__ real sh_vec[];

	real dot = 0;
	real sumexp = 0;

	//probability terms here. 
	sumexp = 0; 
	for (int i = 0; i < num_classes; i ++){
		if (threadIdx.x < cols) sh_vec[threadIdx.x] = weights[i * cols + threadIdx.x];	
		__syncthreads ();

		if ( idx < rows ){
			dot = 0; 
			for (int j = 0; j < cols; j ++) dot += sh_vec[j] * features[ j * rows + idx ]; 	
			sumexp += exp( dot );
		}
		__syncthreads ();
	}

	for (int i = 0; i < num_classes; i ++){
		if (threadIdx.x < cols) sh_vec[threadIdx.x] = weights[i * cols + threadIdx.x];	
		__syncthreads ();

		if ( idx < rows ){
			dot = 0; 
			for (int j = 0; j < cols; j ++) dot += sh_vec[j] * features[ j * rows + idx ]; 	
			B [ idx + i * rows ] = exp(dot) / (1 + sumexp); 
		}
		__syncthreads ();
	}
}

GLOBAL void ker_hx_C_scale (real *A, real *B, real *C, int rows, int cols, int num_classes, real *scale )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	real sum = 0; 
	if (idx < rows){
		for (int i = 0; i < num_classes; i ++) 
			sum += A[ idx + i * rows ] * B[ idx + i * rows ];

		for (int i = 0; i < num_classes; i ++) 
			C[ i * rows + idx ] = 
			 	(1. / scale[ idx ]) * ( A[ idx + i * rows ] * B[ idx + i * rows ] - 
				B[ idx + i * rows ] * sum );
	}
}

GLOBAL void ker_hx_C (real *A, real *B, real *C, int rows, int cols, int num_classes )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	real sum = 0; 
	if (idx < rows){
		for (int i = 0; i < num_classes; i ++) 
			sum += A[ idx + i * rows ] * B[ idx + i * rows ];

		for (int i = 0; i < num_classes; i ++) 
			C[ i * rows + idx ] = 
			 	A[ idx + i * rows ] * B[ idx + i * rows ] - 
				B[ idx + i * rows ] * sum ;
	}
}


real softmax_multiclass_fx (SparseDataset *spfeatures, real *features, real *target, int rows, int cols, int num_classes, 
				real *weights, real lambda, real *devPtr, real *hostPtr, real *pageLckPtr){
	
/*
	ker_softmax <<< BLOCKS, BLOCK_SIZE, sizeof(real) * cols >>> 
		(features, target, rows, cols, num_classes, weights, lambda, devPtr);
	cudaThreadSynchronize ();
	cudaCheckError ();

	reduce <<< 1, BLOCKS_POW_2, BLOCKS_POW_2 * sizeof(real) >>> 
		( devPtr, pageLckPtr, BLOCKS );	
	cudaThreadSynchronize ();
	cudaCheckError ();

        cublasCheckError( cublasDnrm2( cublasHandle, num_classes * cols, weights, 1, &pageLckPtr[1])) ;
	return (pageLckPtr[0]) + (lambda/2.0) * pow(pageLckPtr[1], 2.);
*/	

	
	//matvec operation here. 
        int power = 1;
	real alpha; 
	real beta; 
	real *indicatorVal = devPtr + rows * num_classes; 
	real *maxdots = indicatorVal + rows + BLOCKS_POW_2; 
	real *alphax = maxdots + rows + BLOCKS_POW_2; 
	int NUM_THREADS = 1;

	alpha = 1.0; 
	beta = 0; 
	if (features) {
        	cublasCheckError(cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        rows, num_classes, cols, 
                                        &alpha, features, rows,
                                        weights, cols, &beta, devPtr, rows) );
	} else {
		cusparseCheckError (
			cusparseDcsrmm( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
					rows, num_classes, cols, spfeatures->nnz, 	
					&alpha, spfeatures->descr, spfeatures->sortedVals, spfeatures->rowCsrPtr, 
					spfeatures->colPtr, weights, cols, &beta, devPtr, rows )
			); 
	}
	//fprintf( stderr, "NUM CLASSES --- >%d \n", num_classes ); 
	//fprintf( stderr, "Matvec: \n"); 
	//printVector( devPtr, 20, NULL); 

	ker_compute_fx <<< BLOCKS * NUM_THREADS, BLOCK_SIZE, WARP_SIZE * sizeof(real)  >>> 
			( devPtr, rows, cols, num_classes, target, indicatorVal, NUM_THREADS, maxdots); 
	cudaThreadSynchronize ();  
	cudaCheckError (); 
	//fprintf( stderr, "Exp matvec: ... \n"); 
	//printVector( devPtr, 20, NULL);
	//printVector( maxdots, 20, NULL); 


	//reduce the maxdots here. 
	reduce <<< BLOCKS, BLOCK_SIZE, WARP_SIZE * sizeof (real) >>>
		(maxdots, maxdots + rows, rows ); 
	cudaThreadSynchronize (); 
	cudaCheckError ();	
	//printVector (maxdots + rows, 20, NULL ); 

	reduce <<< 1, BLOCKS_POW_2, WARP_SIZE * sizeof( real ) >>> 
	//reduce <<< 1, WARP_SIZE, WARP_SIZE * sizeof( real ) >>> 
		(maxdots + rows, &pageLckPtr[3], BLOCKS ); 
	cudaThreadSynchronize (); 
	cudaCheckError ();	
	//fprintf( stderr, "Maxdot sum: ... %e \n", pageLckPtr[3]); 
	
	
	// final value of the indicator
	reduce <<< 1, BLOCKS_POW_2, WARP_SIZE * sizeof(real) >>> 
	//reduce <<< 1, WARP_SIZE, WARP_SIZE * sizeof(real) >>> 
		( indicatorVal, &pageLckPtr[0], BLOCKS ); 
	cudaThreadSynchronize (); 
	cudaCheckError ();	

	//fprintf( stderr, "Indicator value: %e \n", pageLckPtr[0] ); 

	/*
	power = 1;
        while (power < num_classes) power *= 2;

	//compute the log par there. 
	reduce_vector_mt <<< THREADS_PER_ROW, WARP_SIZE, WARP_SIZE * sizeof(real) >>> 
		(devPtr, devPtr, rows, 1., num_classes);
	cudaThreadSynchronize (); 
	cudaCheckError ();	
	*/

	//compute the log part here. 
	int warp_blocks = ((rows * WARP_SIZE) / BLOCK_SIZE) + 
				(((rows * WARP_SIZE) % BLOCK_SIZE == 0) ? 0 : 1); 

	//reduce_vector_warp_mt <<< warp_blocks, BLOCK_SIZE  >>> 
	reduce_vector_warp <<< BLOCKS, BLOCK_SIZE >>> 
		(devPtr, maxdots, alphax, rows, num_classes ); 
	cudaThreadSynchronize (); 
	cudaCheckError (); 
	//fprintf( stderr, " Reduce Warp: ....\n"); 
	//printVector( alphax, 20, NULL); 
	

	//final log part here. 
	reduce_log <<< BLOCKS, BLOCK_SIZE, WARP_SIZE* sizeof(real) >>> 
	//reduce <<< BLOCKS, BLOCK_SIZE, WARP_SIZE* sizeof(real) >>> 
		//( devPtr, devPtr, rows ); 
		( alphax, alphax + rows, rows ); 
	cudaThreadSynchronize ();
	cudaCheckError ();
	//printVector( alphax + rows, 20, NULL); 

	reduce <<< 1, BLOCKS_POW_2, WARP_SIZE * sizeof(real) >>> 
	//reduce <<< 1, WARP_SIZE, WARP_SIZE * sizeof(real) >>> 
		//( devPtr, &pageLckPtr[1], BLOCKS);	
		( alphax + rows, &pageLckPtr[1], BLOCKS);	
	cudaThreadSynchronize ();
	cudaCheckError ();
	//fprintf( stderr, "Log part: %e \n", pageLckPtr[1] ); 

	//return pageLckPtr[1]; 

        cublasCheckError( cublasDnrm2( cublasHandle, num_classes * cols, weights, 1, &pageLckPtr[2])) ;
	return (pageLckPtr[3] + pageLckPtr[1]) - pageLckPtr[0] + (lambda/2.0) * pow(pageLckPtr[2], 2.);
	
}

//the result is a vector in here. 
void softmax_multiclass_gx (real *features, real *target, int rows, int cols, 
				int num_classes, real *weights, real lambda, real *gradient, 
				real *devPtr, real *hostPtr, real *pageLckPtr)
{
	ker_dx_softmax <<<BLOCKS, BLOCK_SIZE, cols * sizeof(real) >>> 
		(features, target, rows, cols, num_classes, weights, lambda, devPtr);
	cudaThreadSynchronize (); 
	cudaCheckError ();	

	/*
	reduce_vector <<<1, BLOCKS_POW_2, (BLOCKS_POW_2/WARP_SIZE) * sizeof (real)  >>> 
		(devPtr, gradient, num_classes, cols, 1., BLOCKS );
	*/
	
	//int maxcomps = num_classes * cols + (num_classes * cols) % THREADS_PER_ROW ;
	int maxcomps = num_classes * cols ;
	reduce_vector_mt <<< THREADS_PER_ROW, BLOCKS_POW_2, WARP_SIZE * sizeof(real) >>> 
		(devPtr, gradient, maxcomps, 1., BLOCKS );
	cudaThreadSynchronize (); 
	cudaCheckError ();	

	if (lambda != 0) {
		pageLckPtr[0] = lambda ;
        	cublasCheckError( cublasDaxpy( cublasHandle, num_classes * cols, &pageLckPtr[0], weights, 1, gradient, 1) );
	}
}

// build the hessian here. 
void softmax_multiclass_hx (real *features, int rows, int cols, int num_classes, 
				real *weights, real *vector, real lambda, 
				real *devPtr, real *hostPtr, real *pageLckPtr, real *Hv, real *B, int computeB)
{
	/*
	real *A = devPtr; 
	real *B = A + rows * num_classes;
	real *C = B + rows * num_classes; 
	*/
	real *A = devPtr; 
	real *C = A + rows * num_classes; 

	real *alpha = pageLckPtr; 
	real *beta = alpha + 1;

	//compute A here. 	
	ker_hx_Xv <<< BLOCKS, BLOCK_SIZE, cols * sizeof(real)  >>> 
		(features, vector, rows, cols, num_classes, A); 
	cudaThreadSynchronize ();
	cudaCheckError (); 

	//Compute B Here. 
	if (computeB >= 1) {
		ker_hx_ProbabilityTerms <<<BLOCKS, BLOCK_SIZE, cols * sizeof(real) >>>
		(features, weights, rows, cols, num_classes, B); 
		cudaThreadSynchronize ();
		cudaCheckError (); 
	}
	

	//Compute C Here. 
	ker_hx_C <<< BLOCKS, BLOCK_SIZE >>>
		(A, B, C, rows, cols, num_classes); 
	cudaThreadSynchronize ();
	cudaCheckError (); 

	//Compute the final Matvec Here. 
	*alpha = 1.0; 
	*beta = 0; 
        cublasCheckError(cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                        cols, num_classes, rows,
                                        alpha, features, rows,
                                        C, rows, beta, Hv, cols ) );

/*

	*alpha = 1./(real)(num_classes * rows);
	cublasCheckError (cublasDscal( cublasHandle, num_classes * cols, alpha, Hv, 1) );
*/


	if (lambda != 0) {
		int rblocks = ((num_classes * cols) / BLOCK_SIZE) + 
				(((num_classes * cols) % BLOCK_SIZE == 0) ? 0 : 1 ); 

		//ker_add_regularizer <<< BLOCKS, BLOCK_SIZE >>>
		//(Hv, vector, lambda, num_classes * cols, 1./ (real)rows ); 
		ker_add_regularizer <<< rblocks, BLOCK_SIZE >>>
		(Hv, vector, lambda, num_classes * cols, 1. ); 
		//(Hv, vector, lambda, num_classes * cols, 1./ ((real)rows * num_classes) ); 
		cudaThreadSynchronize (); 
		cudaCheckError ();
	}
}


///////////////////////
//OPTIMIZED CODE HERE
///////////////////////
int generateNonUniformSample( real *probs, real *scaleTerms, int rows, int sampleSize, int *selIndices, real *devPtr, real *hostPtr)
{
        int count = 0;
        real *devIndices = devPtr + rows;

        getRandomVector( rows, NULL, devPtr);

        ker_compute_probs <<< BLOCKS, BLOCK_SIZE >>>
                        ( probs, rows, sampleSize, devPtr, devIndices );
        cudaThreadSynchronize ();
        cudaCheckError ();

        copy_host_device( hostPtr, devIndices, sizeof(real) * rows,
                                                cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST);

        for (int i = 0; i < rows; i ++){
                if (hostPtr[i] != 0)
                        selIndices[ count ++] = i;
        }

        //prepare scaleTerms here. 
        cuda_memset( scaleTerms, 0, sizeof(real) * rows, 0x99 );
        cuda_memset( devIndices, 0, sizeof(real) * rows, 0x99 );
        copy_host_device( selIndices, devIndices, sizeof(int) * count,
                                        cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );

        int blocks = count / BLOCK_SIZE +
                        ((count % BLOCK_SIZE) == 0 ? 0 : 1 );
        ker_init_scaleTerms <<< blocks, BLOCK_SIZE >>>
                        ( scaleTerms, count, probs, (int *)devIndices );
        cudaThreadSynchronize ();
        cudaCheckError ();

        return count;
}

void computeRowProbabilities( SparseDataset *spfeatures, real *features, int rows, int cols, int numclasses,
                        real *dHXW, real *rowNrms, real *probs, real *devPtr )
{
        ker_compute_dHXW_nrm <<< BLOCKS, BLOCK_SIZE >>>
                ( dHXW, rowNrms, rows, numclasses);
        cudaThreadSynchronize ();
        cudaCheckError ();

        //reduce to compute the sum
        reduce <<< BLOCKS, BLOCK_SIZE, WARP_SIZE * sizeof (real) >>>
                (dHXW, devPtr, rows );
        cudaThreadSynchronize ();
        cudaCheckError ();

        reduce <<< 1, BLOCKS_POW_2, WARP_SIZE * sizeof (real) >>>
                (devPtr, devPtr + BLOCK_SIZE, BLOCKS);
        cudaThreadSynchronize ();
        cudaCheckError ();

        ker_normalize <<< BLOCKS, BLOCK_SIZE >>>
                (dHXW, rows, devPtr + BLOCK_SIZE, probs );
        cudaThreadSynchronize ();
        cudaCheckError ();
}

void computeRowNorms( SparseDataset *spfeatures, real *features, int rows, int cols, real *rowNrms, real *devPtr )
{
        if (features != NULL) {
                ker_row_norms <<< BLOCKS, BLOCK_SIZE >>>
                        ( features, rows, cols, rowNrms );
                cudaThreadSynchronize ();
                cudaCheckError ();
        } else {
                cudaMemcpy( spfeatures->valPtr, spfeatures->sortedVals,
                                sizeof(real) * spfeatures->nnz, cudaMemcpyDeviceToDevice );

                int blocks = spfeatures->nnz / (BLOCK_SIZE) +
                                ((spfeatures->nnz % (BLOCK_SIZE)) == 0 ? 0 : 1 );
                ker_sqr_elements <<< blocks, BLOCK_SIZE >>>
                        (spfeatures->valPtr, spfeatures->nnz, 1, devPtr);
                cudaThreadSynchronize ();
                cudaCheckError ();

                //matvec here. for row sums
                real alpha = 1.0;
                real beta = 0;

                //init the vector here. 
                blocks = cols / BLOCK_SIZE + (( cols % BLOCK_SIZE == 0) ? 0 : 1 );
                ker_init_ones <<< blocks, BLOCK_SIZE >>>
                                ( devPtr , cols );
                cudaThreadSynchronize ();
                cudaCheckError ();

                cudaMemset( rowNrms, 0, sizeof(real) * rows );
                cusparseCheckError(
                        cusparseDcsrmv( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        rows, cols, spfeatures->nnz,
                                        &alpha, spfeatures->descr, spfeatures->valPtr, spfeatures->rowCsrPtr,
                                        spfeatures->colPtr, devPtr, &beta, rowNrms)
                                );
                ker_sqrt_elements  <<< BLOCKS, BLOCK_SIZE >>>
                                ( rowNrms, rows);
                cudaThreadSynchronize ();
                cudaCheckError ();
        }
}


void computeHXW (SparseDataset *spfeatures, real *features, int rows, int cols, int num_classes, 
		real *weights, real *XW, int subSampling ) {


	real alpha; 
	real beta; 

	alpha = 1.0; 
	beta = 0; 
	if (features) {
        	cublasCheckError(cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        rows, num_classes, cols, 
                                        &alpha, features, rows,
                                        weights, cols, &beta, XW, rows) );
	} else {
		cusparseCheckError (
			cusparseDcsrmm( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
					rows, num_classes, cols, spfeatures->nnz, 	
					&alpha, spfeatures->descr, spfeatures->sortedVals, spfeatures->rowCsrPtr, 
					spfeatures->colPtr, weights, cols, &beta, XW, rows )
			); 
	}


	if (subSampling >= 1){
		int blocks = rows / BLOCK_SIZE + (((rows % BLOCK_SIZE) == 0) ? 0 : 1 ); 
		ker_compute_HXW <<< blocks, BLOCK_SIZE >>> 
			( XW, rows, cols, num_classes, 1); 
	} else {
		ker_compute_HXW <<< BLOCKS, BLOCK_SIZE >>> 
			( XW, rows, cols, num_classes, 1); 
	}
	cudaThreadSynchronize (); 
	cudaCheckError ();

/*
	ker_hx_ProbabilityTerms <<<BLOCKS, BLOCK_SIZE, cols * sizeof(real) >>>
		(features, weights, rows, cols, num_classes, XW); 
	cudaThreadSynchronize ();
	cudaCheckError (); 
*/
}


void computeExpSum( real *XW, int rows, int cols, int num_classes, real *expSumVec )
{
	ker_compute_expsum <<< BLOCKS, BLOCK_SIZE >>> 
		( XW, rows, cols, num_classes, expSumVec, 1 ); 
	cudaThreadSynchronize (); 
	cudaCheckError ();
}

void softmax_multiclass_gx_subsampled (SparseDataset *spfeatures, real *features, real *target, int rows, int cols, int num_classes, 
			real *weights, real lambda, real *gradient, real *devPtr, real *hostPtr, real *pageLckPtr, 
			SparseDataset *spGradientSample, real *gradientDataset, SparseDataset *spSampledGradientTrain, 
			real *gradientLabels, int sampleSize, int samplingType)
{
	real *HXW = devPtr;
	real *hxwInd = HXW + rows * num_classes; 

	int blocks; 
	real alpha; 
	real beta; 

	//computeHXW Here. 
	computeHXW( spSampledGradientTrain, gradientDataset, sampleSize, cols, num_classes, weights, HXW, 1 ); 

	blocks = sampleSize / BLOCK_SIZE + ((( sampleSize % BLOCK_SIZE ) == 0) ? 0 : 1 ); 
	ker_dx_softmax_ind <<< blocks, BLOCK_SIZE >>> 
			//(HXW, target, sampleSize, num_classes, hxwInd, 1); 
			(HXW, gradientLabels, sampleSize, num_classes, hxwInd, 1); 
	cudaThreadSynchronize (); 
	cudaCheckError ();	

	//compute the gradient here. 
	alpha = 1.0; 
	beta = 0; 

	//perform the X^T * HXWIND
	if (features) {
        	cublasCheckError(cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                        cols, num_classes, sampleSize,
                                        &alpha, gradientDataset, sampleSize,
                                        hxwInd, sampleSize, &beta, gradient, cols ) );
	} else {
		cusparseCheckError( 
			cusparseDcsrmm ( cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
					sampleSize, num_classes, cols, 
					spSampledGradientTrain->nnz, &alpha, 
					spSampledGradientTrain->descr, spSampledGradientTrain->sortedVals, spSampledGradientTrain->rowCsrPtr, 
					spSampledGradientTrain->colPtr, hxwInd, sampleSize, &beta, gradient, cols ) ); 
	}


        //non-uniform subsampling part here. 
	if (samplingType == 2) {
        	alpha = ((real)rows)/((real)sampleSize);
        	cublasCheckError( cublasDscal( cublasHandle, num_classes * cols, &alpha, gradient, 1) );
	} else if (samplingType == 1){
		alpha = ((real) rows) / ((real) sampleSize ); 
		cublasCheckError( cublasDscal( cublasHandle, num_classes * cols, &alpha, gradient, 1 ));
	}

	//regularizer here. 
        cublasCheckError( cublasDaxpy( cublasHandle, num_classes * cols, &lambda, weights, 1, gradient, 1) );
}


void softmax_multiclass_gx_optimized (SparseDataset *spfeatures, real *features, real *target, int rows, int cols, int num_classes, 
			real *weights, real lambda, real *HXW, real *gradient, 
			real *devPtr, real *hostPtr, real *pageLckPtr)
{
	/*
	ker_dx_softmax_mt <<<BLOCKS * THREADS_PER_ROW, BLOCK_SIZE, cols * sizeof(real) >>> 
		(features, target, rows, cols, num_classes, weights, lambda, 
			HXW, expSumVec, devPtr);
	cudaThreadSynchronize (); 
	cudaCheckError ();	

	//reduce across all blocks here. 
	int maxcomps = num_classes * cols ;
	reduce_vector_mt <<< THREADS_PER_ROW, BLOCKS_POW_2, WARP_SIZE * sizeof(real) >>> 
		(devPtr, gradient, maxcomps, 1., BLOCKS );
	cudaThreadSynchronize (); 
	cudaCheckError ();	

	//regularizer here. 
	if (lambda != 0) {
		pageLckPtr[0] = lambda ;
        	cublasCheckError( cublasDaxpy( cublasHandle, num_classes * cols, &pageLckPtr[0], weights, 1, gradient, 1) );
	}
	*/

	cuda_memset( gradient, 0, sizeof(real) * num_classes * cols, ERROR_MEM_SET ); 

	real alpha; 
	real beta; 
	real *hxwInd = devPtr; 

	real dxnrm;
	real gxnrm; 

	ker_dx_softmax_ind <<< BLOCKS , BLOCK_SIZE >>> 
			(HXW, target, rows, num_classes, hxwInd, 1); 

	cudaDeviceSynchronize (); 
	cudaThreadSynchronize (); 
	cudaCheckError ();	

        //cublasCheckError( cublasDnrm2( cublasHandle, rows, hxwInd + rows, 1, &dxnrm));
	//fprintf( stderr, "Norm of the Hxwind matrix is : %f \n", dxnrm * dxnrm ); 
	//printVector( hxwInd, 100, NULL ); 
	//printVector( target, 1000, NULL ); 
	//printVector( target, rows, NULL ); 

	//compute the gradient here. 
	alpha = 1.0; 
	beta = 0; 

	//perform the X^T * HXWIND
	if (features) {
        	cublasCheckError(cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                        cols, num_classes, rows,
                                        &alpha, features, rows,
                                        hxwInd, rows, &beta, gradient, cols ) );
	} else {
		//fprintf( stderr, "Spfeatures nnz: %d \n", spfeatures->nnz ); 
		/*
		cusparseCheckError ( 
			cusparseDcsrmm( cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, 
					rows, num_classes, cols, spfeatures->nnz, 	
					&alpha, spfeatures->descr, spfeatures->valPtr, spfeatures->rowCsrPtr, 
					spfeatures->colPtr, HXW, rows, &beta, gradient, cols) ); 
		*/

		cusparseCheckError( 
			cusparseDcsrmm( cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, 
					rows, num_classes , cols, spfeatures->nnz, 
					&alpha, spfeatures->descr, spfeatures->sortedVals, spfeatures->rowCsrPtr, 
					spfeatures->colPtr, hxwInd, rows, &beta, gradient, cols ) ); 
		
		//sparse matvec here. 
		/*
		cusparseCheckError( 
			cusparseDcsrmv( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
					cols, rows, spfeatures->nnz, 
					&alpha, spfeatures->descr, spfeatures->cscValPtr, spfeatures->cscColPtr, 
					spfeatures->cscRowPtr, hxwInd, &beta, gradient )
				);
			cudaDeviceSynchronize (); 
		*/

		//writeVector( hxwInd, rows, "first_column.txt", 0 ); 
			

		//printVector( gradient, 20, NULL ); 
        	//cublasCheckError( cublasDnrm2( cublasHandle, num_classes * cols, gradient, 1, &gxnrm));
		//fprintf ( stderr, "Gx norm: %f \n", gxnrm ); 
					
	}

	//regularizer here. 
        cublasCheckError( cublasDaxpy( cublasHandle, num_classes * cols, &lambda, weights, 1, gradient, 1) );
}

void softmax_multiclass_hx_subsampled(SparseDataset *spfeatures, real *features, int rows, int cols, int num_classes, 
				real *weights, real *vector, real lambda, 
				real *devPtr, real *hostPtr, real *pageLckPtr, real *Hv, real *HXW, 
				SparseDataset *sampledfeatures, real *sampledDataset, 
				SparseDataset *spSampledHessianTrainSet, int sampleSize, real *scaleTerms, int samplingType)
{
	real *A = devPtr; 
	real *B = A + sampleSize * num_classes; 
	real *C = B + sampleSize * num_classes; 

	real alpha,beta;

	//compute  A = XV
	alpha = 1; 
	beta = 0; 
	if (features) {
        	cublasCheckError(cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        sampleSize, num_classes, cols, 
                                        &alpha, sampledDataset, sampleSize,
                                        vector, cols, &beta, A, sampleSize) );
	} else {
		cusparseCheckError (
			cusparseDcsrmm( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
					sampleSize, num_classes, cols, spSampledHessianTrainSet->nnz, 	
					&alpha, spSampledHessianTrainSet->descr, spSampledHessianTrainSet->sortedVals, 
					spSampledHessianTrainSet->rowCsrPtr, spSampledHessianTrainSet->colPtr, 
					//vector, cols, &beta, A, rows )
					vector, cols, &beta, A, sampleSize)
			); 

		//FIXED-subsampling issue
	}

	//compute B here. for sub sample part of the feautre matrix here. 
	computeHXW( spSampledHessianTrainSet, sampledDataset, sampleSize, cols, num_classes, weights, B, 1 ); 

	//Compute C Here. 
	//ker_hx_C <<< BLOCKS, BLOCK_SIZE >>>
	int blocks = sampleSize / BLOCK_SIZE + (((sampleSize % BLOCK_SIZE) == 0) ? 0 : 1); 
	if (samplingType == 2) {
		ker_hx_C_scale <<< blocks, BLOCK_SIZE >>>
			(A, B, C, sampleSize, cols, num_classes, scaleTerms); 
	} else {
		ker_hx_C <<< blocks, BLOCK_SIZE >>>
			(A, B, C, sampleSize, cols, num_classes); 
	}
	
	cudaThreadSynchronize ();
	cudaCheckError (); 

	//Compute the final Matvec Here. 
	alpha = 1.0; 
	beta = 0; 
	if (features) {
        	cublasCheckError(cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                        cols, num_classes, sampleSize,
                                        &alpha, sampledDataset, sampleSize,
                                        C, sampleSize, &beta, Hv, cols ) );
	} else {
		cusparseCheckError (
			cusparseDcsrmm( cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, 
					sampleSize, num_classes, cols, spSampledHessianTrainSet->nnz, 	
					&alpha, spSampledHessianTrainSet->descr, spSampledHessianTrainSet->sortedVals, 
					spSampledHessianTrainSet->rowCsrPtr, spSampledHessianTrainSet->colPtr, 
					//C, rows, &beta, Hv, cols)
					C, sampleSize, &beta, Hv, cols)
			); 

		//FIXED subsampling issue
	}

	if (samplingType == 1) {
		//scale everything here. 
		alpha = ( ((real) rows) / ((real) sampleSize)); 
        	cublasCheckError( cublasDscal( cublasHandle, num_classes * cols, &alpha, Hv, 1) );
	}

	if (lambda != 0) {
		int rblocks = ((num_classes * cols) / BLOCK_SIZE) + 
				(((num_classes * cols) % BLOCK_SIZE == 0) ? 0 : 1 ); 

		ker_add_regularizer <<< rblocks, BLOCK_SIZE >>>
		(Hv, vector, lambda, num_classes * cols, 1. ); 
		cudaThreadSynchronize (); 
		cudaCheckError ();
	}

}

void softmax_multiclass_hx_optimized (SparseDataset *spfeatures, real *features, int rows, int cols, int num_classes, 
				real *weights, real *vector, real lambda, 
				real *devPtr, real *hostPtr, real *pageLckPtr, real *Hv, real *B )
{
	real *A = devPtr; 
	real *C = A + rows * num_classes; 

	real alpha,beta;

	//compute  A = XV
	alpha = 1; 
	beta = 0; 
	if (features) {
        	cublasCheckError(cublasDgemm( cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        rows, num_classes, cols, 
                                        &alpha, features, rows,
                                        vector, cols, &beta, A, rows) );
	} else {
		cusparseCheckError (
			cusparseDcsrmm( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
					rows, num_classes, cols, spfeatures->nnz, 	
					&alpha, spfeatures->descr, spfeatures->sortedVals, spfeatures->rowCsrPtr, 
					spfeatures->colPtr, vector, cols, &beta, A, rows )
			); 
	}

	//Compute C Here. 
	ker_hx_C <<< BLOCKS, BLOCK_SIZE >>>
		(A, B, C, rows, cols, num_classes); 
	cudaThreadSynchronize ();
	cudaCheckError (); 

	//Compute the final Matvec Here. 
	alpha = 1.0; 
	beta = 0; 
	if (features) {
        	cublasCheckError(cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                        cols, num_classes, rows,
                                        &alpha, features, rows,
                                        C, rows, &beta, Hv, cols ) );
	} else {
		cusparseCheckError (
			cusparseDcsrmm( cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, 
					rows, num_classes, cols, spfeatures->nnz, 	
					&alpha, spfeatures->descr, spfeatures->sortedVals, spfeatures->rowCsrPtr, 
					spfeatures->colPtr, C, rows, &beta, Hv, cols)
			); 
	}

	if (lambda != 0) {
		int rblocks = ((num_classes * cols) / BLOCK_SIZE) + 
				(((num_classes * cols) % BLOCK_SIZE == 0) ? 0 : 1 ); 

		ker_add_regularizer <<< rblocks, BLOCK_SIZE >>>
		(Hv, vector, lambda, num_classes * cols, 1. ); 
		cudaThreadSynchronize (); 
		cudaCheckError ();
	}

}

////////////////////
//DONE HERE
////////////////////



GLOBAL void ker_softmax_predict( real *test_set, real *weights, 
				int rows, int cols, int numclasses, real *workspace)
{
	extern __shared__ real sh_vec[];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	real dot = 0;
	real sumexp; 
	real sumprob; 

	//probability terms here. 
	sumexp = 0; 
	for (int i = 0; i < numclasses; i ++){
		if (threadIdx.x < cols) sh_vec[threadIdx.x] = weights[i * cols + threadIdx.x];	
		__syncthreads ();

		if ( idx < rows ){
			dot = 0; 
			for (int j = 0; j < cols; j ++) dot += sh_vec[j] * test_set[ j * rows + idx ]; 	
			sumexp += exp( dot );
		}
		__syncthreads ();
	}

	for (int c = 0; c < numclasses; c ++) {
		if (threadIdx.x < cols) sh_vec[ threadIdx.x ] = weights[ c * cols + threadIdx.x ];
		__syncthreads ();
		
		if (idx < rows){
			dot = 0.; 
			for (int i = 0; i < cols; i ++) dot += test_set[i * rows + idx] * sh_vec[i];
			workspace[ idx * numclasses + c ] = exp(dot) / (1 + sumexp);
		}
		__syncthreads ();
	}
}

real softmax_predict(SparseDataset *spTest, real *test_set, real *test_labels, real *weights, int rows, int cols, int numclasses, 
			real *hostWorkspace, real *devWorkspace, int computeDevice, real *h_test_set)
{
	int pblocks =  (rows / BLOCK_SIZE) + 
			((rows % BLOCK_SIZE) == 0 ? 0  : 1 );
	real pmax = 0;
	real matches = 0; 
	real nomatches = 0;
	int pclass = -1;
	real sumprob;
	real dot, sumexp, maxdot; 

	real *h_weights = hostWorkspace; 
	real *temp = h_weights + numclasses * cols; 

//	fprintf( stderr, "ROWS -----> %d, COLS --------> %d, CLASSES ------> %d \n", rows, cols, numclasses );

	if (computeDevice == 1) {
		/*
		ker_softmax_predict <<< pblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(real) >>> 
			( test_set, weights, rows, cols, numclasses, devWorkspace);
		cudaThreadSynchronize (); 
		cudaCheckError ();
		*/
		computeHXW( spTest, test_set, rows, cols, numclasses, weights, devWorkspace, 0 ); 

		copy_host_device( temp, devWorkspace, sizeof(real) * numclasses * rows, 
				cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST );
	} else {

		copy_host_device( h_weights, weights, sizeof(real) * numclasses * cols, 
				cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST );

		for (int i = 0; i < rows; i ++) {
			sumexp = 0; 	
			for (int c = 0; c < numclasses; c ++) {
				dot = 0; 
				for (int j = 0; j < cols; j ++) dot += h_test_set[ j * rows + i ] * h_weights[ c * numclasses + j ];
				sumexp += exp ( dot ); 
			}
			sumexp += 1.;

			for (int c = 0; c < numclasses; c ++) {
				dot = 0; 
				for (int j = 0; j < cols; j ++) dot += h_test_set[ j * rows + i ] * h_weights[ c * numclasses + j ];
				temp[ i * numclasses + c ] = exp( dot ) / sumexp; 
			}
		}
	}

#ifdef __debug__
//	fprintf(stderr, " ---------- Class Probabilities ---------\n");
#endif


	// classify here, 
	// Which ever probability is maximum

/*
	int counters[numclasses+1], true_counters[numclasses + 1]; 
	memset( counters, 0, sizeof(int) * (numclasses + 1) ); 
	memset( true_counters, 0, sizeof(int) * (numclasses + 1) ); 
*/


	for (int i = 0; i < rows; i ++){
		
		pmax = 0; 
		pclass = -1;
		sumprob = 0; 
		for (int c = 0; c < numclasses; c ++){

			sumprob += temp[ c * rows + i ];
			if (pmax < temp[ c * rows + i ]){
				pmax = temp[c * rows + i]; 
				pclass = c + 1; 
			}
		}
		
		/*	
		if (pclass < 0) {
			fprintf( stderr, " Error in predicting classes ..... \n"); 
			exit(-1); 
		}
		*/
		
/*
		true_counters[ (int)(test_labels[i]-1) ] ++; 
		if (pmax <= (1.- sumprob))
			counters[numclasses] ++; 
		else
			counters[ pclass - 1 ] ++;
*/


		/*
		if ( ((pmax <= (1. - sumprob)) && (test_labels[i] == (numclasses + 1))) ||
			(pclass == (int)(test_labels[i])) ){
			matches ++; 
		}
		*/
		if ((pmax <= (1. - sumprob)) && (test_labels[i] == (numclasses + 1))){ 
			matches ++; 
		} else if ((pmax > (1. - sumprob)) && (pclass == (int)(test_labels[i])) ) {
			matches ++; 
		} else 
			nomatches ++; 
 		 
		//for (int c = 0; c < numclasses; c ++) fprintf( stderr, " %e ", temp[ c * rows + i] ); 
		//fprintf( stderr, "\n");
	}	


/*

	for (int i = 0; i < numclasses + 1; i ++) 
		fprintf( stderr, " Class: %d ---> Predicted: %d, TrueCount: %d \n", i + 1, counters[i], true_counters[i] );

	fprintf( stderr, "Total matches -----> %f, %d, %f \n", matches, rows, nomatches );
*/
	
	//return ((real)matches/(real)rows) * 100.; 
	return (matches/(matches + nomatches)) * 100.; 
}

void computeErrors ( real *features, real *target, int rows, int cols, int numclasses,
			real *devPtr, real *hostPtr, real *pageLckPtr, int numpoints)
{
	int offset = numclasses * cols % 4; 
	int count; 

	real *constPoint = hostPtr;
	real *hostPoint = constPoint + numclasses * cols + offset; 
	real *dx = hostPoint + numclasses * cols + offset; 
	real *ferror = dx + numclasses * cols + offset;
	real *herror = ferror + numpoints;
	real *dxs = herror + numpoints;
	real *nextHostPtr = dxs + numpoints;

        real *devPoint = devPtr;
	real *devDx = devPoint + numclasses * cols + offset;
        real *gradient = devDx + numclasses * cols + offset;
        real *Hv= gradient + numclasses * cols + offset;
        real *devConstPoint = Hv + numclasses * cols + offset;
	real *B = devConstPoint + numclasses * cols+ offset;

	//real *nextDevPtr = devConstPoint + numclasses * cols + offset; 
	real *nextDevPtr = B+ numclasses * rows+ offset; 

	real *vv = pageLckPtr; 
	real *vhv = vv + 1;
	real *dxnrm = vhv + 1;
	real *nextPagePtr = dxnrm + 1;

	real f;
	real f0;
	real lambda = 0.;
	
	real alpha, beta;

	fprintf( stderr, "Number of random numbers to be generated: %d \n", numclasses * cols );

	memset( constPoint, 0, sizeof(real) * numclasses * cols );
	for (int i = 0; i < numclasses * cols; i ++)	constPoint[i] = 0.;

	copy_host_device( constPoint, devPoint, sizeof(real) * numclasses * cols, 
				cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );
	copy_host_device( constPoint, devConstPoint, sizeof(real) * numclasses * cols, 
				cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );

	getRandomVector( numclasses * cols, dx, nextDevPtr);
	//for (int i = 0; i < numclasses * cols; i ++)	dx[i] = 1.;
	//count = readVector( dx, numclasses * cols, "dx_forest.txt");
	//fprintf( stderr, "Read the random vector from file as: %d \n", count ); 
	
	//printHostVector( dx, numclasses * cols );

	//f0
        f0 = softmax_multiclass_fx (NULL, features, target, rows, cols, numclasses, 
				devPoint, lambda, nextDevPtr, nextHostPtr, nextPagePtr);

	//g0
        softmax_multiclass_gx (features, target, rows, cols,
                                numclasses, devPoint, lambda, gradient,
                                nextDevPtr, nextHostPtr, nextPagePtr);
	fprintf( stderr, "Gradient of the Softmax function is .... \n");
	//printVector( gradient, numclasses * cols, NULL );

/*
	softmax_multiclass_hx (features, rows, cols, numclasses, 
		devConstPoint, devConstPoint, 0, nextDevPtr, nextHostPtr, nextPagePtr, Hv );
	printVector( Hv, numclasses * cols, NULL ); 
*/

	fprintf( stderr, "Starting the derivative test .. %f\n", f0);

        for (int i = 0; i < numpoints; i ++) {

                for (int j = 0; j < numclasses*cols; j ++) hostPoint[j] = constPoint[j] + dx[j];

                copy_host_device( hostPoint, devPoint, sizeof(real) * numclasses * cols,
                                cudaMemcpyHostToDevice, ERROR_MEMCPY_DEVICE_HOST);
		copy_host_device( dx, devDx, sizeof(real) * numclasses * cols, 
					cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );

		//function evaluation here.
        	f = softmax_multiclass_fx (NULL, features, target, rows, cols, numclasses, 
				devPoint, lambda, nextDevPtr, nextHostPtr, nextPagePtr);

		//first order error
        	cublasCheckError( cublasDdot( cublasHandle, numclasses * cols, gradient, 1, devDx, 1, vv) );
		ferror[i] = f - (f0 + *vv);

		//second order error
		softmax_multiclass_hx (features, rows, cols, numclasses, 
					devConstPoint, devDx, 0, nextDevPtr, nextHostPtr, nextPagePtr, Hv, B, 1 );
		*vhv= 0;
        	cublasCheckError( cublasDdot( cublasHandle, numclasses * cols, devDx, 1, Hv, 1, vhv) );

		//herror[i] = f - (f0 + *vv + (0.5 * (*vhv)) / (real)rows );
		herror[i] = f - (f0 + *vv + 0.5 * (*vhv) );
	
		fprintf( stderr, "%d: f --> %e, vv --> %e, vhv--> %e, ferr: %e, herr: %e \n", 
					i, f, *vv, *vhv, ferror[i], herror[i] );

//exit(-1); 
		//dxs here. 
		*dxnrm = 0;
        	cublasCheckError( cublasDnrm2( cublasHandle, numclasses * cols, devDx, 1, dxnrm));
		dxs[i] = *dxnrm;

                for (int j = 0; j < numclasses*cols; j ++) dx[j] = dx[j] / 2.0;
		//break;
	}

	writeVector( ferror, numpoints, "./ferror.txt", 1 ); //host
	writeVector( herror, numpoints, "./herror.txt", 1 ); //host

	//write dx.^2 here
        for (int j = 0; j < numpoints; j ++) hostPtr[j] = pow(dxs[j], 2.); 
	writeVector( constPoint, numpoints, "./dxs_2.txt", 1 ); //host

	//write dx.^3 here
        for (int j = 0; j < numpoints; j ++) hostPtr[j] = pow(dxs[j], 3.); 
	writeVector( constPoint, numpoints, "./dxs_3.txt", 1 ); //host
}


////////////////////////
////HOST Computations Here. 
////////////////////////
real hostFunctionExact( real *features, real *target, real *weights, int numclasses, int rows, int cols)
{
	real logpart = 0; 
	real classpart = 0; 
	real dot, sumexp;

	real maxdot = 0; 
	
	for (int i = 0; i < rows; i ++) {

		sumexp = 0;
		for (int c = 0; c < numclasses; c ++){
			dot = 0; 
			for (int j = 0; j < cols; j ++) 
				dot += features[ j * rows + i ] * weights[ c * cols + j ]; 	

			if (maxdot < dot) maxdot = dot; 

			sumexp += exp( dot );
		} 
		logpart += log( 1 + sumexp );


		int  myclass = (int)(target[ i ] - 1.); 

		dot = 0; 
		if (myclass < numclasses)
			for (int j = 0; j < cols; j ++)
				dot += features[ j * rows + i ] * weights[ myclass * cols + j ];
		
		classpart += dot; 
	}	

	return (logpart - classpart) / ((real) rows); 
}



real hostFunction( real *features, real *target, real *weights, int numclasses, int rows, int cols)
{
	real logpart = 0; 
	real classpart = 0; 
	real dot, alphax, maxdot, sumexp;
	
	for (int i = 0; i < rows; i ++) {

		maxdot = 0; 
		for (int c = 0; c < numclasses; c ++){
			dot = 0; 
			for (int j = 0; j < cols; j ++) 
				dot += features[ j * rows + i ] * weights[ c * cols + j ]; 	

			if (dot > maxdot ) maxdot = dot;
		} 

		sumexp = 0;
		for (int c = 0; c < numclasses; c ++){
			dot = 0; 
			for (int j = 0; j < cols; j ++) 
				dot += features[ j * rows + i ] * weights[ c * cols + j ]; 	

			sumexp += exp( dot - maxdot );
		} 
		alphax = exp( -1. * (maxdot) ) + sumexp; 
		logpart += (maxdot + log( alphax ));


		int  myclass = (int)(target[ i ] - 1.); 

		dot = 0; 
		if (myclass < numclasses)
			for (int j = 0; j < cols; j ++)
				dot += features[ j * rows + i ] * weights[ myclass * cols + j ];
		
		classpart += dot; 
	}	

	//return (logpart - classpart) / ((real) rows); 
	return (logpart - classpart); 
}

void hostGradientExact( real *features, real *target, int numclasses, int rows, int cols, real *weights, real *gradient)
{
	int myclass = 0; 
	real dot = 0, sumexp = 0; 
	real pi;

	memset( gradient, 0, sizeof(real) * numclasses * cols );

	for (int i = 0; i < rows; i ++) {
		myclass = (int)(target[ i ] - 1.); 

		sumexp = 0; 
		for (int c = 0; c < numclasses; c ++){
			dot = 0; 
			for (int j = 0; j < cols; j ++) 
				dot += features[ j * rows + i ] * weights[ c * cols + j ]; 	
			sumexp += exp( dot );
		} 


		for (int c = 0; c < numclasses; c ++){
			pi = 0; 
			for (int j = 0; j < cols; j ++)
				pi += features[ j * rows + i ] * weights[ c * cols + j ];

			pi = exp(pi) / (1 + sumexp);

			for (int j = 0; j < cols; j ++){
				gradient[ c * cols + j ] += (pi - ((myclass == c) ? 1. : 0.)) * features[ j * rows + i ]; 
			}
		}
	}
	for (int i = 0; i < numclasses * cols; i ++) gradient[i] = gradient[i] / ((real) rows);
}

void hostGradient( real *features, real *target, int numclasses, int rows, int cols, real *weights, real *gradient)
{
	int myclass = 0; 
	real dot = 0, maxdot = 0, sumexp = 0, alphax = 0; 
	real pi;

	memset( gradient, 0, sizeof(real) * numclasses * cols );

	for (int i = 0; i < rows; i ++) {
		sumexp = maxdot = alphax = 0;
		for (int c = 0; c < numclasses; c ++){
			dot = 0; 
			for (int j = 0; j < cols; j ++) 
				dot += features[ j * rows + i ] * weights[ c * cols + j ]; 	
			if (dot > maxdot) maxdot = dot; 
		} 

		myclass = (int)(target[ i ] - 1.); 

		sumexp = 0; 
		for (int c = 0; c < numclasses; c ++){
			dot = 0; 
			for (int j = 0; j < cols; j ++) 
				dot += features[ j * rows + i ] * weights[ c * cols + j ]; 	
			sumexp += exp( dot - maxdot);
		} 
		alphax = exp( -1. * (maxdot ) ) + sumexp;


		for (int c = 0; c < numclasses; c ++){
			pi = 0; 
			for (int j = 0; j < cols; j ++)
				pi += features[ j * rows + i ] * weights[ c * cols + j ];

			pi = exp(pi - maxdot) / alphax;

			for (int j = 0; j < cols; j ++){
				gradient[ c * cols + j ] += (pi - ((myclass == c) ? 1. : 0.)) * features[ j * rows + i ]; 
			}
		}
	}	

	//for (int i = 0; i < numclasses * cols; i ++) gradient[i] = gradient[i] / ((real) rows);
}

void computeScale( real *features, real *target, real *weights, int numclasses, int rows, int cols, real *scale, int a, int b)
{
	real sumexp, pa, pb, dot; 
	real maxdot; 

	if (a == b) {
		for (int i = 0; i < rows; i ++) {
			maxdot = 0; 
			for (int c = 0; c < numclasses; c ++) { 
				dot = 0; 
				for (int j = 0; j < cols; j ++) dot += features[ j * rows + i ] * weights[ c * cols + j ];
				if (maxdot < dot) maxdot = dot; 
			}

			sumexp = 0; 
			for (int c = 0; c < numclasses; c ++) { 
				dot = 0; 
				for (int j = 0; j < cols; j ++) dot += features[ j * rows + i ] * weights[ c * cols + j ];
				sumexp += exp( dot - maxdot ); 
			}
			sumexp += exp( -1. * maxdot ); 

			dot = 0; 
			for (int j = 0; j < cols; j ++) dot += features[ j * rows + i ] * weights[ a * cols + j ];

			scale [ i ] = (exp(dot - maxdot) / sumexp) * (1. - (exp(dot - maxdot)/sumexp));
		}
	}	
	else {
		for (int i = 0; i < rows; i ++) {

			maxdot = 0; 
			for (int c = 0; c < numclasses; c ++) { 
				dot = 0; 
				for (int j = 0; j < cols; j ++) dot += features[ j * rows + i ] * weights[ c * cols + j ];
				if (maxdot < dot) maxdot = dot; 
			}

			sumexp = 0; 
			for (int c = 0; c < numclasses; c ++) { 
				dot = 0; 
				for (int j = 0; j < cols; j ++) dot += features[ j * rows + i ] * weights[ c * cols + j ];
				sumexp += exp( dot - maxdot);
			}
			sumexp += exp(-1 * maxdot ); 

			dot = 0; 
			for (int j = 0; j < cols; j ++) dot += features[ j * rows + i ] * weights[ a * cols + j ];

			pa = exp( dot - maxdot) / sumexp; 

			dot = 0; 
			for (int j = 0; j < cols; j ++) dot += features[ j * rows + i ] * weights[ b * cols + j ];
			pb = exp( dot - maxdot) / sumexp; 

			scale[ i ] = -1. * pa * pb; 
		}
	}
}

void computescalex (real *features, real *target, int numclasses, int rows, int cols, real *scale, real *temp )
{
	for (int i = 0; i < rows; i ++)
		for (int j = 0; j < cols; j ++) 
			temp[ j * rows + i ] = scale[ i ] * features[ j * rows + i ];
		
}
void computextscale (real *features, real *target, int numclasses, int rows, int cols, real *temp, real *block )
{
	memset( block, 0, sizeof(real) * cols * cols );

	for (int i = 0; i < cols; i ++){
		for (int j = 0; j < cols; j ++) {
			for (int k = 0; k < rows; k ++) {
				block[ i * cols + j ] += 
					features[i * rows + k] * temp[j * rows + k];
			}
			//block[ i * cols + j ] /= (real) rows;
		}
	}


	//column major * times * column major format here. 
/*
	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < cols; j ++){
			for (int k = 0; k < rows; k ++){
				block[ i * cols + j ] += 
					features[ i * rows + k ] * temp[ j * rows + k ]; 
			}
		}
	}

	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < cols; j ++){
			block[ i * cols + j ] = block[ i * cols + j ] / (real) rows; 
		}
	}
*/
}


void hostHessian( real *features, real *target, int numclasses, int rows, int cols, real *weights, real *hessian, real *s )
{	
	real *scale = s; 
	real *temp = scale + rows;
	real *block = temp + rows * cols; 
	real *offset;

	memset( hessian, 0, sizeof(real) * numclasses * numclasses * cols * cols ); 

	for (int i = 0; i < numclasses; i ++){
		for (int j = 0; j < numclasses; j ++){
			computeScale ( features, target, weights, numclasses, rows, cols, scale, i, j ); 
			//for ( int k = 0; k < rows; k ++) scale[k] = 1.;
			
			computescalex( features, target, numclasses, rows, cols, scale, temp );
			computextscale( features, target, numclasses, rows, cols, temp, block );

			offset = hessian + i * (numclasses * cols) * cols + j * cols;
			for (int k = 0; k < cols; k ++)
				memcpy( offset + k * numclasses * cols, block + k * cols, sizeof(real) * cols ); 
		}
	}
}

void hostHessianVector( real *features, real *target, real *weights, int numclasses, int rows, int cols, 
				real *vector, real *result, real *temp) {
	real *A = temp; 
	real *B = A + rows * numclasses; 
	real *C = B + rows * numclasses; 

	real dot, sumexp, maxdot; 
	real pw, sum; 

	memset( result, 0, sizeof(real) * numclasses * cols );

	//compute A. - stored in column major order
	for (int i = 0; i < rows; i ++){
		for (int c = 0; c < numclasses; c ++){
			dot = 0; 
			for (int j = 0; j < cols; j ++){
				dot += features[ j * rows + i ] * vector[ c * cols + j]; 
			}
			//A[ i * numclasses + c ] = dot; 
			A[ c * rows + i ] = dot; 
		}
	}

	//compute B here. - stored in column major order
	for (int i = 0; i < rows; i ++){
		maxdot = 0; 
		for (int c = 0; c < numclasses; c ++){
			dot = 0; 
			for (int j = 0; j < cols; j ++)
				dot += features[j * rows + i ] * weights[ c * cols + j ];
			
			if (maxdot < dot) maxdot = dot; 
		}

		sumexp = 0; 
		for (int c = 0; c < numclasses; c ++){
			dot = 0; 
			for (int j = 0; j < cols; j ++)
				dot += features[j * rows + i ] * weights[ c * cols + j ];
			
			sumexp += exp( dot - maxdot ); 
		}
		sumexp += exp( -1 * maxdot ); 

		for (int c = 0; c < numclasses; c ++){
			dot = 0; 
			for (int j = 0; j < cols; j ++)
				dot += features[j * rows + i ] * weights[ c * cols + j ];

			pw = exp( dot - maxdot ) / sumexp; 
			//B[ i * numclasses + c ] = pw; 
			B[ c * rows + i ] = pw; 
		}
	}

	//compute C here.  - stored in column major order
	for (int i = 0; i < rows; i ++){
		sum = 0; 
		for (int k = 0; k < numclasses; k ++) 
			//sum += A[ i * numclasses + k ] * B [ i * numclasses + k ];
			sum += A[ k * rows + i ] * B [ k * rows + i ];

		for (int j = 0; j < numclasses; j ++){
			/*
			C[ i * numclasses + j ] = 
				A[ i * numclasses + j ] * B [ i * numclasses + j ] - 
				B[ i * numclasses + j ] * sum; 
			*/
			C[ j * rows + i ] = 
				A[ j * rows + i ] * B [ j * rows + i ] - 
				B[ j * rows + i ] * sum; 
			
		}
	}

	//compute Hessian * vector here. 
/*
	for (int i = 0; i < cols; i ++){
		for (int j = 0; j < numclasses; j ++){
			for (int k = 0; k < rows; k ++) {
				result[ j * cols + i ] += 
					features[ i * rows + k ] * C[k * numclasses + j ];
			}
		}
	}
*/



	//Compute XT * C = stored in column major format
	for (int i = 0; i < cols; i ++){
		for (int j = 0; j < numclasses; j ++){
			for (int k = 0; k < rows; k ++){
				result[ j * cols + i ] += features[ i * rows + k ] * C[ j * rows + k ]; 
			}
		}
	}
}

void hostDerivativeTest ( real *features, real *target, int rows, int cols, int numclasses,
			real *hostPtr, real *devPtr, int numpoints)
{
	int offset = (numclasses * cols) % 4; 

	real *constPoint = hostPtr;
	real *hostPoint = constPoint + numclasses * cols + offset; 
	real *dx = hostPoint + numclasses * cols + offset; 
	real *ferror = dx + numclasses * cols + offset;
	real *dxs = ferror + numpoints;
	real *gradient = dxs + numpoints; 
	real *hessian = gradient + numclasses * cols; 
	real *herror = hessian + numclasses * numclasses * cols * cols ; 
	real *Hv = herror + numpoints; 
	real *hexplicit = Hv + numclasses * cols; 
	real *nextHostPtr = hexplicit + numpoints;

	real f;
	real f0;
	real vv = 0; 
	real vhve, vhv, sum;
	real dxnrm = 0; 

/*
	for (int i = 0; i < cols; i ++)
		for (int j = 0; j < rows; j ++)
			features[i * rows + j ] = i+1; 

	printHostVector( features, 10 ); 
	printHostVector( features + rows, 10 ); 
	printHostVector( features + 2*rows, 10 ); 
*/


	fprintf( stderr, "Number of random numbers to be generated: %d, %d, %d \n", (numclasses) * cols, rows, cols );

	memset( constPoint, 0, sizeof(real) * (numclasses) * cols );
	for (int i = 0; i < (numclasses) * cols; i ++ ) constPoint[i] = 1.0; 
	//getRandomVector((numclasses) * cols, dx, devPtr);
	//for (int i = 0; i < (numclasses) * cols; i ++ ) dx[i] = 1.0; 
	int count = readVector( dx, numclasses * cols, "dx_forest.txt", 0);
	fprintf( stderr, "Total Points read from file: %d \n", count ); 

        f0 = hostFunction(features, target, constPoint, numclasses, rows, cols );
        hostGradient(features, target, numclasses, rows, cols, constPoint, gradient);
	//hostHessian( features, target, numclasses, rows, cols, constPoint, hessian, nextHostPtr);

/*
	fprintf( stderr, "Hessian Matrix.... \n"); 
	for (int i = 0; i < numclasses * cols; i ++){
		for (int j = 0; j < numclasses * cols; j ++) 
			fprintf (stderr, " %e ", hessian[ i * numclasses * cols + j ] ); 
		fprintf (stderr, "\n");
	}

	fprintf( stderr, "Explicit Hessian vecotr product \n"); 
	for (int j = 0; j < numclasses * cols; j ++) {
		sum = 0; 
		for (int k = 0; k < numclasses * cols; k ++)
			sum += hessian[ j * numclasses * cols + k ] * dx[k]; 
		fprintf( stderr, " %e ", sum ); 
	}
	fprintf( stderr, "\n");

	

	hostHessianVector( features, target, constPoint, numclasses, rows, cols, dx, Hv, nextHostPtr );
	fprintf( stderr, "Hessian vecotr product \n"); 
	printHostVector( Hv, numclasses * cols ); 

	exit (-1); 
*/

	fprintf( stderr, " Function at 0: %f \n", f0);
	//printHostVector( gradient, numclasses * cols );


        for (int i = 0; i < numpoints; i ++) {
                for (int j = 0; j < (numclasses)*cols; j ++) hostPoint[j] = constPoint[j] + dx[j];

        	f = hostFunction(features, target, hostPoint, numclasses, rows, cols );

		/*first order error*/
		vv = 0; 
		for (int j = 0; j < (numclasses) * cols; j ++) vv += gradient[j] * dx[j];
		ferror[i] = f - (f0 + vv);

		/* second order error */ 
		vhv = vhve = 0; 
		
		/*
		for (int j = 0; j < numclasses * cols; j ++) {
			sum = 0; 
			for (int k = 0; k < numclasses * cols; k ++)
				sum += hessian[ j * numclasses * cols + k ] * dx[k]; 
			
			//fprintf( stderr, " %e ", sum ); 
			vhve += dx[j] * sum; 	
		}	
		//fprintf( stderr, "\n"); 
		*/
		

		hostHessianVector( features, target, constPoint, numclasses, rows, cols, dx, Hv, nextHostPtr );
		//printHostVector( Hv, numclasses * cols ); 

		//for (int j = 0; j < numclasses * cols ; j ++) vhv += Hv[ j ] * dx[ j ] / (real) rows; 
		for (int j = 0; j < numclasses * cols ; j ++) vhv += Hv[ j ] * dx[ j ]; 
		
		//hexplicit[i] = f - (f0 + vv + 0.5 * vhve);
		herror[i] = f - (f0 + vv + 0.5 * vhv);

		/*dxs here. */
		dxnrm = 0;
		for (int j = 0; j < (numclasses) * cols; j ++) dxnrm += dx[j] * dx[j];
		dxs[i] = sqrt( dxnrm );

                for (int j = 0; j < (numclasses)*cols; j ++) dx[j] = dx[j] / 2.0;

		fprintf( stderr, "%d: f : %e, vv : %e, ferr: %e, dx_2: %e, vhv: %e, herr: %e, dx_3: %e\n", 
				i, f, vv, ferror[i], pow(dxs[i], 2.0), vhv, herror[i], pow(dxs[i], 3.) );
		//fprintf( stderr, "%d: f : %e, vv : %e, ferr: %e, dx_2: %e, vhve: %e, herr: %e, dx_3: %e\n", 
		//		i, f, vv, ferror[i], pow(dxs[i], 2.0), vhve, hexplicit[i], pow(dxs[i], 3.) );

	}

	writeVector( ferror, numpoints, "./ferror.txt", 1 ); /* host */
	writeVector( herror, numpoints, "./herror.txt", 1 ); /* host */
	//writeVector( hexplicit, numpoints, "./hexplicit.txt", 1 ); /* host */

	/* write dx.^2 here */
        for (int j = 0; j < numpoints; j ++) hostPtr[j] = pow(dxs[j], 2.); 
	writeVector( hostPtr, numpoints, "./dxs_2.txt", 1 ); /* host */

        for (int j = 0; j < numpoints; j ++) hostPtr[j] = pow(dxs[j], 3.); 
	writeVector( hostPtr, numpoints, "./dxs_3.txt", 1 ); /* host */
}
