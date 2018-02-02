#include "logistic_fn_indicator.h"
#include "cuda_utils.h"

#include "mat_functions.h"

#include "gen_random.h"
#include "print_utils.h"

#include "classification_kernels.h"

void logistic_fn_indicator (real *features, SparseDataset *spFeatures, real *target, real *weights, real lambda, int rows, int cols, real *fn, real *devPtr, real *hostPtr)
{
	//host
	real *alpha = hostPtr;
	real *beta = alpha + 1;
	real *nrm_weights = beta + 1;

	//device
	real *t = devPtr;
	real *out = t + rows;	
	real *redResult = out + rows; 

	//features * weights
	*alpha = 1;
	*beta = 0;

	if (spFeatures->valPtr == NULL) {
		cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_N, rows, cols,
				alpha, features, rows, 
				weights, 1, 
				beta, t, 1) );
	} else {
		cusparseCheckError( 
			cusparseDcsrmv( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
				rows, cols, spFeatures->nnz, 	
				alpha, spFeatures->descr, spFeatures->sortedVals, spFeatures->rowCsrPtr, spFeatures->colPtr, 
				weights, beta, t ) ); 
	}

/*
	fprintf( stderr, "printing t    " ); 
	printVector( t + rows - 1, 1, NULL); 
	fprintf( stderr, "printing target    " ); 
	printVector( target + rows - 1, 1, NULL ); 
	fprintf( stderr, "printing out    " ); 
	printVector( out + rows - 1,1, NULL ); 
*/

	ker_log_sum <<<BLOCKS, BLOCK_SIZE >>> ( t, target, rows, out);
	cudaThreadSynchronize ();
	cudaCheckError ();

	ker_reduction <<< BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(real) >>> (out, redResult, rows);
	cudaThreadSynchronize ();
	cudaCheckError ();

	ker_reduction <<< 1, BLOCKS_POW_2, BLOCKS_POW_2 * sizeof(real) >>> (redResult, fn, BLOCKS);
	cudaThreadSynchronize ();
	cudaCheckError ();

	//add the regularization term here. 
	cublasCheckError( cublasDnrm2( cublasHandle, cols, weights, 1, nrm_weights) ); 

	//since we are minimizing this function here. 
	(*fn) += pow(*nrm_weights, 2.) * (lambda/2.0);
}

// sigma( x_ij( y_i - g(z_i))	
// g(z_i) = sigmoid( x_ij * w_i )

void logistic_fn_indicator_gx (real *features, SparseDataset *spFeatures, real *target, real *weights, real lambda, int rows, int cols, real *gn, real *devPtr, real *hostPtr, int samplingType, int numFeatures)
{
	//device
	real *t = devPtr;

	//host
	real *alpha = hostPtr;
	real *beta = alpha + 1;

	//blocks
	int numBlocks = BLOCKS; 
	if (samplingType != 0)
		numBlocks = rows / BLOCK_SIZE + ((rows % BLOCK_SIZE) == 0 ? 0 : 1); 

	*alpha = 1;
	*beta = 0;
	if (spFeatures->valPtr == NULL) {	
		cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_N, rows, cols,
				alpha, features, rows, 
				weights, 1, 
				beta, t, 1) );
	} else {
		cusparseCheckError( 
			cusparseDcsrmv( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
				rows, cols, spFeatures->nnz, 	
				alpha, spFeatures->descr, spFeatures->sortedVals, spFeatures->rowCsrPtr, spFeatures->colPtr, 
				weights, beta, t ) ); 
	}
		
	ker_sigmoid_target <<<numBlocks, BLOCK_SIZE >>> (t, target, rows, t);
	cudaThreadSynchronize ();
	cudaCheckError ();
	
	*alpha = 1;
	*beta = 0; 
	if (spFeatures->valPtr == NULL) {
		cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_T, rows, cols, 
				alpha, features, rows, 
				t, 1, 
				beta, gn, 1) );
	} else {
		cusparseCheckError( 
			cusparseDcsrmv( cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, 
				rows, cols, spFeatures->nnz, 	
				alpha, spFeatures->descr, spFeatures->sortedVals, spFeatures->rowCsrPtr, spFeatures->colPtr, 
				t, beta, gn ) ); 
	}


	// non uniform sampling scaling 
        *alpha = ((real)numFeatures)/((real)rows);
	if (samplingType == 2) {
        	cublasCheckError( cublasDscal( cublasHandle, cols, alpha, gn, 1) );
   	} else if (samplingType ==1 ){
		cublasCheckError( cublasDscal( cublasHandle, cols, alpha, gn, 1) ); 
	}

	//regularization here. 
	*alpha = lambda;
	cublasCheckError( cublasDaxpy( cublasHandle, cols, alpha, weights, 1, gn, 1 ) );
}

GLOBAL void ker_hx_C_scale (real *A, real *B, real *C, int rows, real *scale )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < rows){
		C[ idx ] = (1. / scale[ idx ]) * ( A[ idx ] * B[ idx ] - B[ idx ] * ( A[ idx ] * B[ idx ] ) );
	}
}

GLOBAL void ker_hx_C (real *A, real *B, real *C, int rows )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < rows){
		C[ idx ] = A[ idx ] * B[ idx ] - B[ idx ] * ( A[ idx ] * B[ idx ] );
	}
}

void logistic_fn_indicator_hx_matvec (real *features, SparseDataset *spFeatures, real *weights, real *vector, 
				real lambda, int rows, int cols, real *hx, real *devPtr, real *hostPtr, int samplingType, real *scaleTerms, int numFeatures)
{
	real *A = devPtr; 
	real *B = A + rows; 
	real *C = B + rows; 

	real alpha, beta; 

	//blocks
	int numBlocks = BLOCKS; 
	if (samplingType != 0)
		numBlocks = rows / BLOCK_SIZE + ((rows % BLOCK_SIZE) == 0 ? 0 : 1); 

	//compute A = matrix * vector
	alpha = 1; 
	beta = 0; 
	if (spFeatures->valPtr == NULL) {
		cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_N, rows, cols, 
				&alpha, features, rows, 
				vector, 1, 
				&beta, A, 1) );
	} else {
		cusparseCheckError( 
			cusparseDcsrmv( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
				rows, cols, spFeatures->nnz, 	
				&alpha, spFeatures->descr, spFeatures->sortedVals, spFeatures->rowCsrPtr, spFeatures->colPtr, 
				vector, &beta, A ) ); 
	}

	//compute B = Probability Matrix here. matrix * weights
	if (spFeatures->valPtr == NULL) {
		cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_N, rows, cols, 
				&alpha, features, rows, 
				weights, 1, 
				&beta, B, 1) );
	} else {
		cusparseCheckError( 
			cusparseDcsrmv( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
				rows, cols, spFeatures->nnz, 	
				&alpha, spFeatures->descr, spFeatures->sortedVals, spFeatures->rowCsrPtr, spFeatures->colPtr, 
				weights, &beta, B ) ); 
	}

	ker_sigmoid<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(real) >>> 
		(B, rows, B);
	cudaThreadSynchronize ();
	cudaCheckError ();

	//Compute C = A.B - B.(A.B)
	if (samplingType == 2) {
        ker_hx_C_scale <<< numBlocks, BLOCK_SIZE >>> (A, B, C, rows, scaleTerms);
	} else { 
        ker_hx_C<<< numBlocks, BLOCK_SIZE >>> (A, B, C, rows);
	}
        cudaThreadSynchronize ();
        cudaCheckError ();

	//compute X^T * C = matvec	
	if (spFeatures->valPtr == NULL) {
		alpha = 1.0; 
		beta = 0; 
		cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_T, rows, cols, 
				&alpha, features, rows, 
				C, 1, 
				&beta, hx, 1) );
	} else {
		cusparseCheckError( 
			cusparseDcsrmv( cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, 
				rows, cols, spFeatures->nnz, 	
				&alpha, spFeatures->descr, spFeatures->sortedVals, spFeatures->rowCsrPtr, spFeatures->colPtr, 
				C, &beta, hx ) ); 
	}

	//appropriate scaling
	if (samplingType == 1){ 
		alpha =  ((real)numFeatures/ ((real) rows)); 
		cublasCheckError ( cublasDscal( cublasHandle, cols, &alpha, hx, 1 ) );
	}

	//regularization here. 
	//this is a matrix operation. 
	int colBlockSize = BLOCK_SIZE; 
	int colBlocks = (cols % colBlockSize) == 0 ? (cols/colBlockSize) : (cols/colBlockSize + 1);
	ker_hx_matvec_reg <<<colBlocks, colBlockSize>>>
			(hx, lambda, vector, cols);	
	cudaThreadSynchronize ();
	cudaCheckError ();

}


	
void logistic_fn_indicator_hx (real *features, SparseDataset *spFeatures, real *target, real *weights, real lambda, int rows, int cols, real *hx, real *devPtr, real *hostPtr)
{
	//device
	real *t = devPtr;
	real *t_minus = t + rows;
	real *C = t_minus + rows;

	//host
	real *alpha = hostPtr;
	real *beta = alpha + 1;

	*alpha = 1;
	*beta = 0;
	if ( spFeatures == NULL) {
		cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_N, rows, cols,
				alpha, features, rows, 
				weights, 1, 
				beta, t, 1 ) );
	} else {
		cusparseCheckError( 
			cusparseDcsrmv( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
				rows, cols, spFeatures->nnz, 	
				alpha, spFeatures->descr, spFeatures->sortedVals, spFeatures->rowCsrPtr, spFeatures->colPtr, 
				weights, beta, t ) ); 
	}
	cublasCheckError( cublasDcopy( cublasHandle, rows, t, 1, t_minus, 1 ) );

	//apply sigmoid here. 
	ker_sigmoid<<<BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(real) >>> 
			(t, rows, t);
	cudaThreadSynchronize ();
	cudaCheckError ();

	//fprintf( stderr, "Output from the sigmoid function \n");
	//printVector( t, rows, NULL);
		
	*alpha = -1;
	cublasCheckError ( cublasDscal( cublasHandle, rows, alpha, t_minus, 1 ) );
	ker_sigmoid<<<BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(real) >>> 
			(t_minus, rows, t_minus);
	cudaThreadSynchronize ();
	cudaCheckError ();

	//fprintf( stderr, "Output from the sigmoid function -t \n");
	//printVector( t_minus, rows, NULL);

	//element wise product of two vectors here. 
	ker_ele_vec_product <<< BLOCKS, BLOCK_SIZE >>>
			( t, t_minus, rows, t );
	cudaThreadSynchronize ();
	cudaCheckError ();
	//fprintf( stderr, "Output from the ele vector product\n");
	//printVector( t, rows, NULL);

	// perform the final mat * mat product here. 
	// perform diag(s * neg_s) * features. 
	if (spFeatures == NULL ){ 
		cublasCheckError (cublasDdgmm( cublasHandle, CUBLAS_SIDE_LEFT, 
					rows, cols, features, rows, 
					t, 1, 				
					C, rows) );
		//perform the first. product( features^T x above_result);
		*alpha = 1;
		*beta = 0;
		cublasCheckError(cublasDgemm( cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
					cols, cols, rows, 
					alpha, features, rows, 
					C, rows, beta, hx, cols ) );			
	} else {
		//Not implemented here. 
		//since we are using matvec here for Hessian
		;
	}

	//regularization here. 
	//this is a matrix operation. 
	int colBlockSize = BLOCK_SIZE; 
	int colBlocks = (cols % colBlockSize) == 0 ? colBlocks = cols/colBlockSize : colBlocks = cols/colBlockSize + 1;
	//fprintf ( stderr, "Regularization BLOCKS --> %d and BlockSize -- > %d \n", colBlocks, colBlockSize );
	ker_mat_identity <<<colBlocks, colBlockSize>>>
			(hx, lambda, cols);	
			//(hx, 2 * (lambda), cols);	
	cudaThreadSynchronize ();
	cudaCheckError ();
}

///////////////////////////////////
//Non uniform subsampling code here. 
///////////////////////////////////

int generateNonUniformSample_log( real *probs, real *scaleTerms, int rows, int sampleSize, int *selIndices, real *devPtr, real *hostPtr)
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

//fprintf( stderr, "selected points for non uniform sampling is %d \n", count ); 
        
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

void computeRowProbabilities_log( SparseDataset *spfeatures, real *features, int rows, int cols,
                        real *dHXW, real *rowNrms, real *probs, real *devPtr )
{
        ker_compute_dHXW_nrm_log <<< BLOCKS, BLOCK_SIZE >>>
                ( dHXW, rowNrms, rows);
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


void computeRowNorms_log( SparseDataset *spfeatures, real *features, int rows, int cols, real *rowNrms, real *devPtr )
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

void computeHXW_log (SparseDataset *spfeatures, real *features, int rows, int cols, real *weights, real *B) {
   real alpha; 
   real beta; 

   alpha = 1.0; 
   beta = 0; 

   if (spfeatures->valPtr == NULL) { 
      cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_N, rows, cols,
            &alpha, features, rows,
            weights, 1,  
            &beta, B, 1) );
   } else {
      cusparseCheckError(
         cusparseDcsrmv( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            rows, cols, spfeatures->nnz,
            &alpha, spfeatures->descr, spfeatures->sortedVals, spfeatures->rowCsrPtr, spfeatures->colPtr,
            weights, &beta, B ) );
   }

   //ker_sigmoid<<<BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(real) >>>
   ker_sigmoid<<<BLOCKS, BLOCK_SIZE>>>
      (B, rows, B);
   cudaThreadSynchronize ();
   cudaCheckError ();
}


      


//
//
// PREDICTION HERE. For the Logistic Regression with Indicator random variable
//			as the class label
//
void logistic_regression_predict( real *features, SparseDataset *spFeatures, real *weights, real *labels, real *hostLabels, int rows, int cols, real *accuracy, real *devPtr, real *hostPtr )
{
	real alpha, beta;
	real *sigmoid_predictions = devPtr;
	real nrm;
	int counter0, counter1;

	alpha = 1; 
	beta = 0;
	if (spFeatures->valPtr == NULL) {
		cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_N, rows, cols,  
				&alpha, features, rows, 
				weights, 1, 
				&beta, sigmoid_predictions, 1 ) );
	} else {
		cusparseCheckError( 
			cusparseDcsrmv( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
				rows, cols, spFeatures->nnz, 	
				&alpha, spFeatures->descr, spFeatures->sortedVals, spFeatures->rowCsrPtr, spFeatures->colPtr, 
				weights, &beta, sigmoid_predictions ) ); 
	}

	//apply the sigmoid function here. 
	int tblocks;
	if (rows <= BLOCK_SIZE) 
		tblocks = 1;
	else 
		tblocks = (rows % BLOCK_SIZE) == 0 ? rows / BLOCK_SIZE : (rows/BLOCK_SIZE) + 1;

	//ker_sigmoid_classify<<<tblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(real) >>> (sigmoid_predictions, rows);
	ker_sigmoid_classify<<<tblocks, BLOCK_SIZE >>> (sigmoid_predictions, rows);
	cudaThreadSynchronize ();
	cudaCheckError ();

	copy_host_device( hostPtr, sigmoid_predictions, sizeof(real) * rows, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST );

	*accuracy = 0;
	counter0 = counter1 = 0; 
	for (int i = 0; i < rows; i ++) 
	{	
		if (hostPtr[i] == (hostLabels[i] - 1.0)) (*accuracy) ++;
		if (hostPtr[i] == 1.) counter1 ++; 
		if (hostPtr[i] == 0.) counter0 ++;
	}
	//fprintf( stderr, "0: %d, 1: %d \n", counter0, counter1 ); 

	*accuracy = ((*accuracy) / rows) * 100.;
}



////////////////////////////////////////////////////////
//Derivative Test
////////////////////////////////////////////////////////
/*
void getRandomVectorLogistic (int n, real *hostPtr, real *devPtr) {

        curandGenerator_t gen ;
        int m = n + n % 2;

        curandCheckError ( curandCreateGenerator (&gen , CURAND_RNG_PSEUDO_DEFAULT ) );

        curandCheckError ( curandSetPseudoRandomGeneratorSeed ( gen , 1234ULL )) ;

        curandCheckError ( curandGenerateNormalDouble ( gen , devPtr , m, 0, .25)) ;
        //curandCheckError ( curandGenerateUniformDouble ( gen , devPtr , m)) ;

        copy_host_device( hostPtr, devPtr, sizeof(real) * m, cudaMemcpyDeviceToHost,
                                ERROR_MEMCPY_DEVICE_HOST );

        curandCheckError ( curandDestroyGenerator ( gen ) );
}
*/


void logisticRegDerivativeTest ( real *features, real *target, int rows, int cols, 
                        real *devPtr, real *hostPtr, real *pageLckPtr, int numpoints)
{
        int offset = cols % 4;

        real *constPoint = hostPtr;
        real *hostPoint = constPoint + cols + offset;
        real *dx = hostPoint + cols + offset;
        real *ferror = dx + cols + offset;
        real *herror = ferror + numpoints;
        real *dxs = herror + numpoints;
        real *nextHostPtr = dxs + numpoints;

        real *devPoint = devPtr;
        real *devDx = devPoint + cols + offset;
        real *gradient = devDx + cols + offset;
        real *hessian = gradient + cols + offset;
        real *nextDevPtr = hessian + cols * cols + offset;

        real *vv = pageLckPtr;
        real *vhv = vv + 1;
        real *dxnrm = vhv + 1;
	real *f = dxnrm + 1;
	real *f0 = f + 1;
        real *nextPagePtr = f0 + 1;

        real alpha, beta;

        fprintf( stderr, "Number of random numbers to be generated: %d \n", cols );

        memset( constPoint, 0, sizeof(real) * cols );
        for (int i = 0; i < cols; i ++)  constPoint[i] = 0.;

        copy_host_device( constPoint, devPoint, sizeof(real) * cols,
                                cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );

        //getRandomVectorLogistic( cols, dx, nextDevPtr);
        getRandomVector( cols, dx, nextDevPtr);
        //for (int i = 0; i < cols; i ++)  dx[i] = 0;

        //printHostVector( dx, cols );

        //f0
	//logistic_fn_indicator( features, target, devPoint, 0, rows, cols, f0, nextDevPtr, nextHostPtr);

        //g0
	//logistic_fn_indicator_gx( features, NULL, target, devPoint, 0, rows, cols, gradient, nextDevPtr, nextHostPtr); 
        //printVector( gradient, 5, NULL );

        //h0
	//logistic_fn_indicator_hx( features, target, devPoint, 0, rows, cols, hessian, nextDevPtr, nextHostPtr );

        fprintf( stderr, "Starting the derivative test .. %f\n", *f0);

        for (int i = 0; i < numpoints; i ++) {

                for (int j = 0; j < cols; j ++) hostPoint[j] = constPoint[j] + dx[j];

                copy_host_device( hostPoint, devPoint, sizeof(real) * cols,
                                cudaMemcpyHostToDevice, ERROR_MEMCPY_DEVICE_HOST);
                copy_host_device( dx, devDx, sizeof(real) * cols,
                                        cudaMemcpyHostToDevice, ERROR_MEMCPY_HOST_DEVICE );

                //function evaluation here.
		//logistic_fn_indicator( features, target, devPoint, 0, rows, cols, f, nextDevPtr, nextHostPtr);

                //first order error
                //printVector( gradient, 5, NULL );
                //printVector( devPoint, 5, NULL );
                //fprintf( stderr, "Gradient sum: %e \n", computeWeightSum( gradient, cols ));
                cublasCheckError( cublasDdot( cublasHandle, cols, gradient, 1, devDx, 1, vv) );
                ferror[i] = (*f - (*f0 + *vv)) / (real)rows;

                //second order error
                alpha = 1;
                beta = 0;
                cublasCheckError( cublasDgemv( cublasHandle, CUBLAS_OP_N, cols, cols,
                                        &alpha, hessian, cols,
                                        devDx, 1,
                                        &beta, nextDevPtr, 1) );
                *vhv= 0;
                cublasCheckError( cublasDdot( cublasHandle, cols, devDx, 1, nextDevPtr, 1, vhv) );

                herror[i] = (*f - (*f0 + *vv + 0.5 * (*vhv) )) / (real) rows;

                fprintf( stderr, "%d: f --> %e, vv --> %e, vhv--> %e, ferr: %e, herr: %e \n",
                                        i, *f, *vv, *vhv, ferror[i], herror[i] );

                //dxs here. 
                *dxnrm = 0;
                cublasCheckError( cublasDnrm2( cublasHandle, cols, devDx, 1, dxnrm));
                dxs[i] = *dxnrm;
		//printVector( devDx, 10, NULL);
		//fprintf( stderr, "DevDx norm is ----> %e, %e, %e \n", *dxnrm, pow( *dxnrm, 2.), pow(*dxnrm, 3.) );

                for (int j = 0; j < cols; j ++) dx[j] = dx[j] / 2.0;
                //break;
        }

        writeVector( ferror, numpoints, "./ferror.txt", 1 ); //host
        writeVector( herror, numpoints, "./herror.txt", 1 ); //host

        //write dx.^2 here
        for (int j = 0; j < numpoints; j ++) constPoint[j] = pow(dxs[j], 2.);
        writeVector( constPoint, numpoints, "./dxs_2.txt", 1 ); //host

        //write dx.^3 here
        for (int j = 0; j < numpoints; j ++) constPoint[j] = pow(dxs[j], 3.);
        writeVector( constPoint, numpoints, "./dxs_3.txt", 1 ); //host
}


