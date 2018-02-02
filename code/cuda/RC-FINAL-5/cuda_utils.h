#ifndef	__CUDA_UTILS_H_
#define __CUDA_UTILS_H_

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"
#include "stdlib.h"
#include "stdio.h"
#include "curand.h"


void cuda_malloc (void **, unsigned int , int , int);
void cuda_malloc_host( void **, unsigned int, int, int );

void cuda_free (void *, int);
void cuda_free_host (void *, int);
void cuda_memset (void *, int , size_t , int );

void copy_host_device (void *, void *, int , enum cudaMemcpyKind, int);
void copy_device (void *, void *, int , int );

void print_device_mem_usage ();

#define cusparseCheckError(cusparseStatus) __cusparseCheckError (cusparseStatus, __FILE__, __LINE__)
inline void __cusparseCheckError( cusparseStatus_t cusparseStatus, const char *file, const int line )
{
if (cusparseStatus!= CUSPARSE_STATUS_SUCCESS)
{
	//fprintf (stderr, "failed .. %s:%d -- error code %d \n", __FILE__, __LINE__, cusparseStatus);
	fprintf (stderr, "failed .. %s:%d -- error code %d \n", file, line, cusparseStatus);
	exit (-1);
}
return;
}


#define cublasCheckError(cublasStatus) __cublasCheckError (cublasStatus, __FILE__, __LINE__)
inline void __cublasCheckError( cublasStatus_t cublasStatus, const char *file, const int line )
{
if (cublasStatus!= CUBLAS_STATUS_SUCCESS)
{
	fprintf (stderr, "failed .. %s:%d -- error code %d \n", file, line, cublasStatus);
	exit (-1);
}
return;
}

#define cudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line )
{
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		fprintf (stderr, "Failed .. %s:%d -- gpu erro code %d:%s\n", file, line, err, cudaGetErrorString( err ) );
		exit( -1 );
	}
 
	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	/*
	err = cudaDeviceSynchronize();
	if( cudaSuccess != err )
	{
		exit( -1 );
	}
	*/
	return;
}

#define curandCheckError(curandStatus) __curandCheckError (curandStatus, __FILE__, __LINE__)
inline void __curandCheckError( curandStatus_t curandStatus, const char *file, const int line )
{
        if (curandStatus!= CURAND_STATUS_SUCCESS)
        {
                fprintf (stderr, "failed .. %s:%d -- error code %d \n", __FILE__, __LINE__, curandStatus);
                exit (-1);
        }
        return;
}



void compute_blocks ( int *blocks, int *block_size, int count );
void compute_nearest_pow_2 (int blocks, int *result);


#endif
