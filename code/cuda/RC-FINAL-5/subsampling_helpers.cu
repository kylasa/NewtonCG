
#include <subsampling_helpers.h> 

#include "cuda_utils.h"
#include "print_utils.h"
#include "gen_random.h"

GLOBAL void kerInitSampleMatrix( int *row, int *col, real *val, real *labels, real *srcLabels, int count, int offset, int maxRows )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (idx < count) {
		row[ idx ] = idx; 
		val[ idx ] = 1.; 

		//reshuffle the labels here. 	
		labels[ idx ] = srcLabels[ col[ idx ] ] ; 
	}
}

GLOBAL void kerInitSampleMatrixNoLabels( int *row, int *col, real *val, int count, int offset, int maxRows )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (idx < count) {
		row[ idx ] = idx; 
		val[ idx ] = 1.; 
	}
}

void initSubSampledHessian( int offset, int rows, SparseDataset *sampledSet, real *sampledLabels, real *srcLabels, int sampledSize ){

	int blocks = (sampledSize / BLOCK_SIZE) + 
			(((sampledSize % BLOCK_SIZE) == 0) ? 0 : 1) ;

	if (sampledLabels == NULL && srcLabels == NULL){
		kerInitSampleMatrixNoLabels <<< blocks, BLOCK_SIZE >>> 
			(sampledSet->rowPtr, sampledSet->colPtr, sampledSet->valPtr, sampledSize, offset, rows ); 
	} else {
		kerInitSampleMatrix <<< blocks, BLOCK_SIZE >>> 
			(sampledSet->rowPtr, sampledSet->colPtr, sampledSet->valPtr, sampledLabels, srcLabels, 
				sampledSize, offset, rows ); 
	}
	cudaThreadSynchronize (); 
	cudaCheckError (); 
}

void prepareForNonUniformSampling (SparseDataset *samplingMat, int sampleSize, int *indices) {

        copy_host_device( indices, samplingMat->colPtr, sizeof(int) * sampleSize,
                        cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );
        initSubSampledHessian( -1, -1, samplingMat, NULL, NULL, sampleSize);
}

void prepareForSampling (SparseDataset *sampledGradient, real *sampledLabels, real *srcLabels, int rows, int sampleSize, int *hostPtr) {

        int startRow = -1;

	//generate random rows here for sampling. 
	//genRandomVector( hostPtr, sampleSize, rows ); 	
	genRandomVector( hostPtr, sampleSize, rows - 1 ); 	

	copy_host_device( hostPtr, sampledGradient->colPtr, sizeof(int) * sampleSize, 
			cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE ); 	

        startRow = rand () % rows;
        initSubSampledHessian( startRow, rows, sampledGradient, sampledLabels, srcLabels, sampleSize);
}

void sampleDataset ( SparseDataset *spSampledGradient, real *dataset, 
			int rows, int cols, int num_classes, 
			real *subSampledGradient, int sampleSize )
{
	real alpha = 1.0; 
	real beta = 0; 

	cusparseCheckError (
        	cusparseDcsrmm( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                	sampleSize, cols, rows, spSampledGradient->nnz,
                        &alpha, spSampledGradient->descr, spSampledGradient->sortedVals, spSampledGradient->rowCsrPtr,
                        spSampledGradient->colPtr, dataset, rows, &beta, subSampledGradient, sampleSize)
                        );
}

void sampleSparseDataset ( SparseDataset *spSampler, SparseDataset *spDataset, 
				int rows, int cols, int num_classes, 
				SparseDataset *spGradientSample, int sampleSize )
{
	int *nnzHostPtr = &spGradientSample->nnz; 
	int baseC = 0; 

	cusparseCheckError( 
		cusparseSetPointerMode( cusparseHandle, CUSPARSE_POINTER_MODE_HOST) );
	
	cusparseCheckError (
		cusparseXcsrgemmNnz( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
			//sampleSize, cols, sampleSize, 
			sampleSize, cols, rows, 
			spSampler->descr, spSampler->nnz, spSampler->rowCsrPtr, spSampler->colPtr, 
			spDataset->descr, spDataset->nnz, spDataset->rowCsrPtr, spDataset->colPtr, 
			spGradientSample->descr, spGradientSample->rowCsrPtr, nnzHostPtr
			) ); 

	if (nnzHostPtr != NULL){
		spGradientSample->nnz = *nnzHostPtr; 
	} else {
		cudaMemcpy( &spGradientSample->nnz, spGradientSample->rowCsrPtr + sampleSize, sizeof(int), 
					cudaMemcpyDeviceToHost ); 
		cudaMemcpy( &baseC, spGradientSample->rowCsrPtr, sizeof(int), cudaMemcpyDeviceToHost ); 
		
		spGradientSample->nnz -= baseC; 
	}

	cusparseCheckError (
		cusparseDcsrgemm( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
			//sampleSize, cols, sampleSize, 
			sampleSize, cols, rows, 
			spSampler->descr, spSampler->nnz, spSampler->sortedVals, spSampler->rowCsrPtr, spSampler->colPtr, 
			spDataset->descr, spDataset->nnz, spDataset->sortedVals, spDataset->rowCsrPtr, spDataset->colPtr, 
			spGradientSample->descr, spGradientSample->sortedVals, 
				spGradientSample->rowCsrPtr, spGradientSample->colPtr ) ); 
}
