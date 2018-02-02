#include "cuda_types.h"
#include "conjugate_gradient.h"
#include "cuda_utils.h"
#include "print_utils.h"

#include "softmax_multiclass.h"
#include "logistic_fn_indicator.h"

#include "float.h"
#include "time.h"
#include "stdlib.h"
#include "subsampling_helpers.h"
#include "sparse_dataset.h"

int Cublas_CG_Logistic( DeviceDataset *data, NEWTON_CG_PARAMS *params, 
		real *p_gradient, real *x, real *x_best, real *rel_residual, 
		real *devPtr, real *hostPtr, real *pgeLckPtr)
{
	real *Hg, *residual, *p, *gradient;
	real *rsold, *rsnew, *alpha, *tmp;
	real *nextHostPtr, *nextDevPtr; 
	real gradient_norm;
	real best_rel_residual;

	real *B, *probs, *scaleTerms, *rowNrms; 
	int *selIndices;

	int i;

	Hg = devPtr;
	residual = Hg + data->cols;
	p = residual + data->cols;
	gradient = p + data->cols;

   B = gradient + data->cols;    
   probs = B + data->rows;
   scaleTerms = probs + data->rows;
   rowNrms = scaleTerms + data->rows;
   nextDevPtr = rowNrms + data->rows;
	

	rsold = pgeLckPtr;
	rsnew = &pgeLckPtr[1];
	alpha = &pgeLckPtr[2];
	tmp = &pgeLckPtr[3];

   selIndices = (int *)hostPtr;
   nextHostPtr = hostPtr + data->rows;


	//Perform the sampling for Hessian here. 
	if (params->hx_sampling >= 1) {	

		data->hessianSampleSize = (HESSIAN_SAMPLING_SIZE * data->rows) / 100; 

		prepareForSampling( &data->spHessianSample, NULL, NULL, data->rows, data->hessianSampleSize, (int *)nextHostPtr );
		data->spHessianSample.nnz = data->hessianSampleSize; 

		//sample Hessian Here. 
		if (data->spTrain.valPtr == NULL) { 
			//Dense Case
			convertHessianSampleToCSR( &data->spHessianSample, data->hessianSampleSize, data->cols, nextDevPtr ); 
			sampleDataset (&data->spHessianSample, data->trainSet, data->rows, data->cols, data->numclasses, data->sampledHessianTrainSet, data->hessianSampleSize); 
		} else { 
			//Sparse Case
			convertHessianSampleToCSR( &data->spHessianSample, data->hessianSampleSize, data->cols, nextDevPtr ); 
			sampleSparseDataset( &data->spHessianSample, &data->spTrain, data->rows, data->cols, data->numclasses, 
					&data->spSampledHessianTrain, data->hessianSampleSize ); 
		}

		logistic_fn_indicator_hx_matvec( data->sampledHessianTrainSet, &data->spSampledHessianTrain, data->weights, x, params->lambda, data->hessianSampleSize, data->cols, Hg, nextDevPtr, nextHostPtr, params->hx_sampling, scaleTerms, data->rows ); 
	}
	else {
		logistic_fn_indicator_hx_matvec( data->trainSet, &data->spTrain, data->weights, x, params->lambda, data->rows, data->cols, Hg, nextDevPtr, nextHostPtr, params->hx_sampling, scaleTerms, data->rows ); 
	}

	*alpha = -1;
	cublasCheckError (cublasDcopy( cublasHandle, data->cols, p_gradient, 1, gradient, 1) );
	cublasCheckError (cublasDscal( cublasHandle, data->cols, alpha, gradient, 1) );


	// residual = g - H*g;
	cublasCheckError (cublasDcopy( cublasHandle, data->cols, gradient, 1, residual, 1) );
	*alpha = -1;
	cublasCheckError (cublasDaxpy( cublasHandle, data->cols, alpha, Hg, 1, residual, 1 ) );

	//p = residual;
	cublasCheckError (cublasDcopy( cublasHandle, data->cols, residual, 1, p, 1) );

	//rsold = Dot( residual, residual, N );
	cublasCheckError (cublasDdot( cublasHandle, data->cols, residual, 1, residual, 1, rsold ) ); 
	
	cublasCheckError( cublasDnrm2( cublasHandle, data->cols, gradient, 1, &gradient_norm) ); 
	best_rel_residual = SQRT( *rsold ) / gradient_norm; 
	cudaMemcpy( x_best, x, data->cols * sizeof(real), cudaMemcpyDeviceToDevice ); 

	for( i = 0; i < params->max_cg_iterations; ++i ) {
		//hessian vec here
		if (params->hx_sampling > 0) {
			logistic_fn_indicator_hx_matvec( data->sampledHessianTrainSet, &data->spSampledHessianTrain, 
							data->weights, p, params->lambda, data->hessianSampleSize, data->cols, Hg, 
							nextDevPtr, nextHostPtr, params->hx_sampling, scaleTerms, data->rows ); 
		} else {
			logistic_fn_indicator_hx_matvec( data->trainSet, &data->spTrain, 
							data->weights, p, params->lambda, data->rows, data->cols, Hg, 
							nextDevPtr, nextHostPtr, params->hx_sampling, scaleTerms, data->rows ); 
		}

		//tmp = Dot( Hg, p, N );
		cublasCheckError (cublasDdot( cublasHandle, data->cols, Hg, 1, p, 1, tmp ) ); 
		*alpha = -1. * ((*rsold) / (*tmp));    

		//Vector_Add( residual, -alpha, Hg, N ); //residual = residual - alpha * Hg
		cublasCheckError (cublasDaxpy( cublasHandle, data->cols, alpha, Hg, 1, residual, 1 ) );

		*alpha *= -1.;
		//Vector_Add( x, alpha, p ); x = x + alpha * p
		cublasCheckError (cublasDaxpy( cublasHandle, data->cols, alpha, p, 1, x, 1 ) );

		//rsnew = Dot (residual, residual);
		cublasCheckError (cublasDdot( cublasHandle, data->cols, residual, 1, residual, 1, rsnew ) );

		*rel_residual =  SQRT( *rsnew ) / gradient_norm; 

		if (*rel_residual < best_rel_residual) {
			best_rel_residual = *rel_residual; 
			cudaMemcpy( x_best, x, data->cols * sizeof(real), cudaMemcpyDeviceToDevice ); 
		}
		if (*rel_residual <= params->cg_tolerance) break;

		//p = residual + (rsnew / rsold) * p;
		*alpha = (*rsnew/(*rsold));
		cublasCheckError (cublasDscal( cublasHandle, data->cols, alpha, p, 1) );

		*alpha = 1;
		cublasCheckError (cublasDaxpy( cublasHandle, data->cols, alpha, residual, 1, p, 1 ) );
	 	*rsold = *rsnew;
  	}

	*rel_residual = best_rel_residual;

	return i;
}

int Cublas_CG_multi_optimized(SparseDataset *spfeatures,  real *features, real *g, real *weights, 
				real *x, real *x_best, real lambda, int rows, int cols, int numclasses, real *HXW, 
				real *devPtr, real *hostPtr, real *pgeLckPtr, int MAX_ITERATIONS, 
				real tolerance, real *rel_residual, real *best_rel_residual, 
				SparseDataset *spSampledHessian, real *sampledHessian, 
				SparseDataset *spSampledHessianTrainSet, int hessianSampleSize, int samplingType )
{

	//CG local's Here
	real *p, *r, *h, *alpha, *pAp;
	real  rnorm, gradient_norm, tol2, delta, bb, prev_delta; 
	int iter;

	//Other Locals Here
	real *Hg, *B;
	real *nextDevPtr, *nextHostPtr, *nextPageLckPtr; 

   int *selIndices, nonUniformSampleSize, sampleSize;
   real *rowNrms, *probs, *scaleTerms;

	//Device Pointers
	Hg = devPtr;
	r = Hg + numclasses * cols;
	p = r + numclasses * cols;
	B = p + numclasses * cols; 	
   probs = B + rows * numclasses;
   scaleTerms = probs + rows;
   rowNrms = scaleTerms + rows;
	h = rowNrms + numclasses * cols;	
   nextDevPtr = h + rows;
	
	//PageLock Pointers
	alpha = &pgeLckPtr[0];
	pAp = &pgeLckPtr[1];
	nextPageLckPtr = pAp + 1; 

	//Host Only Pointers
   selIndices = (int *)hostPtr;
   nextHostPtr = hostPtr + rows;


	//Initializations here. 
	sampleSize = hessianSampleSize; 

	if (samplingType >= 1) {	

		if (samplingType == 1) { 
			sampleSize = hessianSampleSize; 
			prepareForSampling( spSampledHessian, NULL, NULL, rows, sampleSize, (int *)nextHostPtr );
		} else {
    		computeHXW( spfeatures, features, rows, cols, numclasses, weights, B, 0 );
      	computeRowNorms( spfeatures, features, rows, cols, rowNrms, nextDevPtr );
      	computeRowProbabilities( spfeatures, features, rows, cols, numclasses, B, rowNrms, probs, nextDevPtr );
      	nonUniformSampleSize = generateNonUniformSample( probs, scaleTerms, rows, hessianSampleSize, selIndices, nextDevPtr, nextHostPtr );

			sampleSize = nonUniformSampleSize; 
        	prepareForNonUniformSampling( spSampledHessian, sampleSize, selIndices );
		}
		spSampledHessian->nnz = sampleSize; 

		//sample Hessian Here. 
		if (features) { 
			convertHessianSampleToCSR( spSampledHessian, sampleSize, cols, nextDevPtr ); 
			sampleDataset (spSampledHessian, features, rows, cols, numclasses, sampledHessian, sampleSize ); 
		} else { 
			convertHessianSampleToCSR( spSampledHessian, sampleSize, cols, nextDevPtr ); 
			sampleSparseDataset( spSampledHessian, spfeatures, rows, cols, numclasses, 
					spSampledHessianTrainSet, sampleSize ); 
		}

		softmax_multiclass_hx_subsampled(spfeatures,  features, rows, cols, numclasses, 
			weights, x, lambda, nextDevPtr, nextHostPtr, nextPageLckPtr, Hg, HXW,
			spSampledHessian, sampledHessian, spSampledHessianTrainSet, sampleSize, scaleTerms, samplingType ); 
	}
	else {
      softmax_multiclass_hx_optimized(spfeatures,  features, rows, cols, numclasses,
      	weights, x, lambda, nextDevPtr, nextHostPtr, nextPageLckPtr, Hg, HXW );
	}
	

	//tol2 = tol^2
	tol2 = pow( tolerance, 2. );

	// r = g - H*g;
	cublasCheckError (cublasDcopy( cublasHandle, numclasses * cols, g, 1, r, 1) );
	*alpha = -1;
	cublasCheckError (cublasDaxpy( cublasHandle, numclasses * cols, alpha, Hg, 1, r, 1 ) );

	//h = Precondition( P, r)
	cublasCheckError( cublasDcopy( cublasHandle, numclasses * cols, r, 1, h, 1) );

	//delta = r' * h
	cublasCheckError( cublasDdot( cublasHandle, numclasses * cols, r, 1, h, 1, &delta ) );

	//bb = b' * Preconditioned( P, b)
	cublasCheckError( cublasDdot( cublasHandle, numclasses * cols, g, 1, g, 1, &bb ) );

	//p = r;
	cublasCheckError (cublasDcopy( cublasHandle, numclasses * cols, r, 1, p, 1) );

	//Store the best result to return
	*best_rel_residual = DBL_MAX;
	cudaMemcpy( x_best, x, numclasses * cols * sizeof(real), cudaMemcpyDeviceToDevice ); 

	iter = 0;
	cublasCheckError( cublasDnrm2( cublasHandle, numclasses * cols, g, 1, &gradient_norm) ); 
	cublasCheckError( cublasDnrm2( cublasHandle, numclasses * cols, r, 1, &rnorm) ); 
	*rel_residual =  rnorm / gradient_norm; 

	while ( (delta > tol2 * bb) && (iter < MAX_ITERATIONS) && (*rel_residual > tolerance) ) {

		if (samplingType != 0) {
			softmax_multiclass_hx_subsampled(spfeatures,  features, rows, cols, numclasses, 
				weights, p, lambda, nextDevPtr, nextHostPtr, nextPageLckPtr, Hg, HXW, 
				spSampledHessian, sampledHessian, spSampledHessianTrainSet, sampleSize, scaleTerms, samplingType ); 
		}
		else {
         softmax_multiclass_hx_optimized(spfeatures,  features, rows, cols, numclasses,
         	weights, p, lambda, nextDevPtr, nextHostPtr, nextPageLckPtr, Hg, HXW );
		}

		//pAp = Dot( Hg, p, N );
		cublasCheckError (cublasDdot( cublasHandle, numclasses * cols, Hg, 1, p, 1, pAp ) ); 

		//alpha = delta / pAp
		*alpha = -1. * (delta / (*pAp) );    

		//r = r - alpha * Ap
		cublasCheckError (cublasDaxpy( cublasHandle, numclasses * cols, alpha, Hg, 1, r, 1 ) );

		// x = x + alpha * p
		*alpha *= -1.;
		cublasCheckError (cublasDaxpy( cublasHandle, numclasses * cols, alpha, p, 1, x, 1 ) );

		// rel_res = norm(r) / norm(b)
		cublasCheckError (cublasDnrm2( cublasHandle, numclasses * cols, r, 1, &rnorm) );
		*rel_residual =  rnorm / gradient_norm; 

		if (*rel_residual < *best_rel_residual) {
			*best_rel_residual = *rel_residual; 
			cudaMemcpy( x_best, x, numclasses * cols * sizeof(real), cudaMemcpyDeviceToDevice ); 
		}

		//h = r
		cublasCheckError( cublasDcopy( cublasHandle, numclasses * cols, r, 1, h, 1) );

		prev_delta = delta; 

		//delta = r' * h
		cublasCheckError( cublasDdot( cublasHandle, numclasses * cols, r, 1, h, 1, &delta) ); 

		//p = h + (delta/prev_delta) * p;
		*alpha = delta / prev_delta;
		cublasCheckError( cublasDscal( cublasHandle, numclasses * cols, alpha, p, 1) );
		*alpha = 1; 
		cublasCheckError( cublasDaxpy( cublasHandle, numclasses * cols, alpha, h, 1, p, 1) ); 

		//increment the iteration count here
		cublasCheckError( cublasDnrm2( cublasHandle, numclasses * cols, r, 1, &rnorm) ); 
		*rel_residual =  rnorm / gradient_norm; 
		iter += 1;
  	}

	return iter;
}
