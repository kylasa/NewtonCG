
#include "classification_kernels.h"

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

GLOBAL void ker_compute_dHXW_nrm_log (real *dHXW, real *rowNrms, int rows)
{
        int myRowId = blockIdx.x * blockDim.x + threadIdx.x;

        if (myRowId < rows) {
                dHXW[ myRowId ] = abs( dHXW[ myRowId ] * (1. - dHXW[ myRowId ]) ) * rowNrms[ myRowId ];
        }
}


GLOBAL void ker_normalize (real *dHXW, int rows, real *nrmConstant, real *probs ){
        int myRowId = blockIdx.x * blockDim.x + threadIdx.x;
        if (myRowId < rows){
                probs[ myRowId ] = dHXW[ myRowId ] / nrmConstant[0];
        }
}

GLOBAL void ker_row_norms( real *features, int rows, int cols, real *nrm )
{
        int myRowId = ( blockIdx.x * blockDim.x + threadIdx.x );
        int i = 0;
        real sum = 0;

        if (myRowId < rows) {
                i = myRowId;
                for (int j = 0; j < cols; j += 1)
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
