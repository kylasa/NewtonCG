
#include "cuda_types.h"
#include "cuda_utils.h"
#include "sparse_dataset.h"

void initMatDescriptors( DeviceDataset *d )
{
	//Train
	cusparseCheckError ( cusparseCreateMatDescr(&(d->spTrain.descr)) ); 
	cusparseCheckError ( cusparseSetMatIndexBase(d->spTrain.descr, CUSPARSE_INDEX_BASE_ZERO) );
	cusparseCheckError ( cusparseSetMatType(d->spTrain.descr, CUSPARSE_MATRIX_TYPE_GENERAL) );

	//Test
	cusparseCheckError ( cusparseCreateMatDescr(&(d->spTest.descr)) ); 
	cusparseCheckError ( cusparseSetMatIndexBase(d->spTest.descr, CUSPARSE_INDEX_BASE_ZERO) );
	cusparseCheckError ( cusparseSetMatType(d->spTest.descr, CUSPARSE_MATRIX_TYPE_GENERAL) );
}

void initMatDescriptorsForSampling( DeviceDataset *d ) {

	//SubSampling - Hessian
	cusparseCheckError ( cusparseCreateMatDescr(&(d->spHessianSample.descr)) ); 
	cusparseCheckError ( cusparseSetMatIndexBase(d->spHessianSample.descr, CUSPARSE_INDEX_BASE_ZERO) );
	cusparseCheckError ( cusparseSetMatType(d->spHessianSample.descr, CUSPARSE_MATRIX_TYPE_GENERAL) );

	//gradient
	cusparseCheckError ( cusparseCreateMatDescr(&(d->spGradientSample.descr)) ); 
	cusparseCheckError ( cusparseSetMatIndexBase(d->spGradientSample.descr, CUSPARSE_INDEX_BASE_ZERO) );
	cusparseCheckError ( cusparseSetMatType(d->spGradientSample.descr, CUSPARSE_MATRIX_TYPE_GENERAL) );
}

void initMatDescriptorsForSparseSampling( DeviceDataset *d ) {

	//SubSampling - Hessian
	cusparseCheckError ( cusparseCreateMatDescr(&(d->spSampledHessianTrain.descr)) ); 
	cusparseCheckError ( cusparseSetMatIndexBase(d->spSampledHessianTrain.descr, CUSPARSE_INDEX_BASE_ZERO) );
	cusparseCheckError ( cusparseSetMatType(d->spSampledHessianTrain.descr, CUSPARSE_MATRIX_TYPE_GENERAL) );

	//gradient
	cusparseCheckError ( cusparseCreateMatDescr(&(d->spSampledGradientTrain.descr)) ); 
	cusparseCheckError ( cusparseSetMatIndexBase(d->spSampledGradientTrain.descr, CUSPARSE_INDEX_BASE_ZERO) );
	cusparseCheckError ( cusparseSetMatType(d->spSampledGradientTrain.descr, CUSPARSE_MATRIX_TYPE_GENERAL) );
}

void convertGradientSampleToCSR (SparseDataset *spGradientSample, int sampleSize, int cols, real *devPtr) {

	//make sure that the data is sorted here. 
	size_t pBufferSizeInBytes = 0; 
	void* pBuffer = (void *)devPtr; 

	//Sampled Dataset Here. 
	cusparseCheckError( 
			cusparseXcoosort_bufferSizeExt( 
				cusparseHandle, sampleSize, cols, spGradientSample->nnz, 
				spGradientSample->rowPtr, spGradientSample->colPtr, &pBufferSizeInBytes ) ); 

	cusparseCheckError( 
		cusparseCreateIdentityPermutation( cusparseHandle, spGradientSample->nnz, spGradientSample->P) ); 
	
	cusparseCheckError( 
		cusparseXcoosortByRow( cusparseHandle, sampleSize, cols, spGradientSample->nnz, 
				spGradientSample->rowPtr, spGradientSample->colPtr, spGradientSample->P, pBuffer ) ); 

	cusparseCheckError( 
		cusparseDgthr( cusparseHandle, spGradientSample->nnz, spGradientSample->valPtr, 
				spGradientSample->sortedVals, spGradientSample->P, CUSPARSE_INDEX_BASE_ZERO ) ); 

	//convert to csr format. 
	cusparseCheckError( 
			cusparseXcoo2csr( cusparseHandle, spGradientSample->rowPtr, spGradientSample->nnz, sampleSize, 
				spGradientSample->rowCsrPtr, CUSPARSE_INDEX_BASE_ZERO ) 
		); 	

	//fprintf( stderr, "Converting gradient to CSR .... \n"); 
}


void convertHessianSampleToCSR (SparseDataset *spHessianSample, int sampleSize, int cols, real *devPtr) {

	//make sure that the data is sorted here. 
	size_t pBufferSizeInBytes = 0; 
	void* pBuffer = (void *)devPtr; 

	//Sampled Dataset Here. 
	cusparseCheckError( 
			cusparseXcoosort_bufferSizeExt( 
				cusparseHandle, sampleSize, cols, spHessianSample->nnz, 
				spHessianSample->rowPtr, spHessianSample->colPtr, &pBufferSizeInBytes ) ); 

	cusparseCheckError( 
		cusparseCreateIdentityPermutation( cusparseHandle, spHessianSample->nnz, spHessianSample->P) ); 
	
	cusparseCheckError( 
		cusparseXcoosortByRow( cusparseHandle, sampleSize, cols, spHessianSample->nnz, 
				spHessianSample->rowPtr, spHessianSample->colPtr, spHessianSample->P, pBuffer ) ); 

	cusparseCheckError( 
		cusparseDgthr( cusparseHandle, spHessianSample->nnz, spHessianSample->valPtr, 
				spHessianSample->sortedVals, spHessianSample->P, CUSPARSE_INDEX_BASE_ZERO ) ); 

	//convert to csr format. 
	cusparseCheckError( 
			cusparseXcoo2csr( cusparseHandle, spHessianSample->rowPtr, spHessianSample->nnz, sampleSize, 
				spHessianSample->rowCsrPtr, CUSPARSE_INDEX_BASE_ZERO ) 
		); 	

	//fprintf( stderr, "Converting hessian to CSR .... \n"); 
}

void convertToCSR( DeviceDataset *d, real *devPtr )
{
	//make sure that the data is sorted here. 
	size_t pBufferSizeInBytes = 0; 
	void* pBuffer = (void *)devPtr; 

	//Train Dataset Here. 
	cusparseCheckError( 
			cusparseXcoosort_bufferSizeExt( 
				cusparseHandle, d->rows, d->cols, d->spTrain.nnz, 
				d->spTrain.rowPtr, d->spTrain.colPtr, &pBufferSizeInBytes ) ); 
	fprintf( stderr, "Memory needed to sort coo data --> %d \n", pBufferSizeInBytes ); 

	cusparseCheckError( 
		cusparseCreateIdentityPermutation( cusparseHandle, d->spTrain.nnz, d->spTrain.P) ); 
	
	cusparseCheckError( 
		cusparseXcoosortByRow( cusparseHandle, d->rows, d->cols, d->spTrain.nnz, 
				d->spTrain.rowPtr, d->spTrain.colPtr, d->spTrain.P, pBuffer ) ); 

	cusparseCheckError( 
		cusparseDgthr( cusparseHandle, d->spTrain.nnz, d->spTrain.valPtr, 
				d->spTrain.sortedVals, d->spTrain.P, CUSPARSE_INDEX_BASE_ZERO ) ); 

	//convert to csr format. 
	cusparseCheckError( 
			cusparseXcoo2csr( cusparseHandle, d->spTrain.rowPtr, d->spTrain.nnz, d->rows, 
				d->spTrain.rowCsrPtr, CUSPARSE_INDEX_BASE_ZERO ) 
		); 	


	//Test Dataset here. 
	cusparseCheckError( 
			cusparseXcoosort_bufferSizeExt( 
				cusparseHandle, d->rows, d->cols, d->spTest.nnz, 
				d->spTest.rowPtr, d->spTest.colPtr, &pBufferSizeInBytes ) ); 
	fprintf( stderr, "Memory needed to sort coo data --> %d \n", pBufferSizeInBytes ); 

	cusparseCheckError( 
		cusparseCreateIdentityPermutation( cusparseHandle, d->spTest.nnz, d->spTest.P) ); 
	
	cusparseCheckError( 
		cusparseXcoosortByRow( cusparseHandle, d->rows, d->cols, d->spTest.nnz, 
				d->spTest.rowPtr, d->spTest.colPtr, d->spTest.P, pBuffer ) ); 

	cusparseCheckError( 
		cusparseDgthr( cusparseHandle, d->spTest.nnz, d->spTest.valPtr, 
				d->spTest.sortedVals, d->spTest.P, CUSPARSE_INDEX_BASE_ZERO ) ); 

	//convert to csr format. 
	cusparseCheckError( 
			cusparseXcoo2csr( cusparseHandle, d->spTest.rowPtr, d->spTest.nnz, d->rows, 
				d->spTest.rowCsrPtr, CUSPARSE_INDEX_BASE_ZERO ) 
		); 	

/*
	cusparseCheckError( 
			cusparseXcoo2csr( cusparseHandle, d->spTest.rowPtr, d->spTest.nnz, d->testSize, 
				d->spTest.rowCsrPtr, CUSPARSE_INDEX_BASE_ZERO ) 
		); 	

	//convert the csr matrix to csc matrix here. 
	cusparseCheckError( 
			cusparseDcsr2csc( cusparseHandle, d->rows, d->cols, d->spTrain.nnz, 
					d->spTrain.valPtr, d->spTrain.rowCsrPtr, d->spTrain.colPtr, 
					d->spTrain.cscValPtr, d->spTrain.cscRowPtr, d->spTrain.cscColPtr, 
					CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO ) ); 
*/
}
