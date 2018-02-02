#ifndef _H_CUDA_TYPES__
#define _H_CUDA_TYPES__

#include "cuda.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "host_defines.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define HOST __host__
#define DEVICE __device__
#define GLOBAL __global__
#define HOST_DEVICE __host__ __device__


#define real double
#define SQRT sqrt

#define HOST_WORKSPACE_SIZE		((1 * 1024 * 1024 * 1024) + (512 * 1024 * 1024))
//#define DEVICE_WORKSPACE_SIZE		((1 * 1024 * 1024 * 1024) + (512 * 1024 * 1024))
#define DEVICE_WORKSPACE_SIZE		1 * 1024 * 1024 * 1024
#define PAGE_LOCKED_WORKSPACE_SIZE	1024 * 1024

#define DEBUG_SCRATCH_SIZE		10 * 1024 * 1024

#define ERROR_MEM_ALLOC			0x01
#define ERROR_MEM_CLEANUP		0x02
#define ERROR_MEMCPY_DEVICE_HOST	0x03

#define ERR_MEM_ALLOC			0x04
#define ERR_MEM_FREE			0x05

#define ERROR_MEMCPY_TRAINSET		0x06
#define ERROR_MEMCPY_TESTSET		0x07
#define ERROR_MEMCPY_TRAINLABELS 	0x08
#define ERROR_MEMCPY_TESTLABELS 	0x09

#define ERROR_DEBUG			0x10
#define ERROR_MEM_SET			0x11

#define ERROR_MEMCPY_DEVICE_DEVICE	0x12
#define ERROR_MEMCPY_HOST_DEVICE	0x13

#define CUDA_BLOCK_SIZE			1024

#define WARP_SIZE			32
#define THREADS_PER_ROW			64


//#define HESSIAN_SAMPLING_SIZE		1
//#define GRADIENT_SAMPLING_SIZE		5
//#define HESSIAN_SAMPLING_SIZE		25
//#define GRADIENT_SAMPLING_SIZE		50

extern int BLOCKS, BLOCK_SIZE, BLOCKS_POW_2;
extern int HESSIAN_SAMPLING_SIZE, GRADIENT_SAMPLING_SIZE; 


extern cublasHandle_t cublasHandle;
extern cusparseHandle_t cusparseHandle; 
typedef struct scratch_space{ 
	real *hostWorkspace; 
	real *devWorkspace;
	real *pageLckWorkspace;
	} SCRATCH_AREA;

extern void* dscratch;

#endif
