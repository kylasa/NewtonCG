#include "linesearch.h"
#include "logistic_fn_indicator.h"
#include "cuda_utils.h"
#include "print_utils.h"

#include "softmax_multiclass.h"

real cg_linesearch (real *d, real *weights, real rho, real c, SparseDataset *spfeatures, real *features, real *target, 
			real lambda, int rows, int cols, int numclasses, real *gk, real *xx, real *devPtr, real *hostPtr, real *pageLocked)
{
	real alphak = 1.; 
	real temp;
	real *fk = &pageLocked[0];
	real *fk1 = &pageLocked[1];
	real *nextPagePtr = pageLocked + 2;

	real *x = devPtr; 
	real *nextDevPtr = x + numclasses * cols;
	int iterations = 0; 

	cublasCheckError( cublasDcopy( cublasHandle, numclasses * cols, weights, 1, x, 1) );

	/*
	if (numclasses == 1)
        	logistic_fn_indicator( features, spfeatures, target, x, lambda, rows, cols, fk, nextDevPtr, hostPtr);	
	else 
	*/
		*fk = softmax_multiclass_fx (spfeatures, features, target, rows, cols, numclasses, x,
                                lambda, nextDevPtr, hostPtr, nextPagePtr);
//fprintf (stderr, "%e, %d, %d, %d\n", *fk, rows, cols, numclasses ); 

	//xx = x;	
	cublasCheckError( cublasDcopy( cublasHandle, numclasses * cols, x, 1, xx, 1) );

	//x = x + alphak*d
	cublasCheckError( cublasDaxpy( cublasHandle, numclasses * cols, &alphak, d, 1, x, 1) );

	cublasCheckError( cublasDnrm2( cublasHandle, numclasses * cols, d, 1, &temp )) ;
	
	/*
	if (numclasses == 1)
        	logistic_fn_indicator( features, spfeatures, target, x, lambda, rows, cols, fk1, nextDevPtr, hostPtr);	
	else
	*/
		*fk1 = softmax_multiclass_fx (spfeatures, features, target, rows, cols, numclasses, x,
                                lambda, nextDevPtr, hostPtr, nextPagePtr);
//fprintf (stderr, "%e, %d, %d, %d\n", *fk1, rows, cols, numclasses ); 

	cublasCheckError( cublasDdot( cublasHandle, numclasses * cols, gk, 1, d, 1, &temp) );
	while (((*fk1) > ((*fk) + c * alphak * temp)) && (iterations < 50)){
		alphak *= rho;

		cublasCheckError( cublasDcopy( cublasHandle, numclasses * cols, xx, 1, x, 1) );
		cublasCheckError( cublasDaxpy( cublasHandle, numclasses * cols, &alphak, d, 1, x, 1) );

		/*
		if (numclasses == 1)
        		logistic_fn_indicator( features, spfeatures, target, x, lambda, rows, cols, fk1, nextDevPtr, hostPtr);	
		else 
		*/
			*fk1 = softmax_multiclass_fx (spfeatures, features, target, rows, cols, numclasses, x,
                                lambda, nextDevPtr, hostPtr, nextPagePtr);

		iterations ++; 
//fprintf (stderr, "%e, %d, %d, %d\n", *fk1, rows, cols, numclasses ); 
	}
	//fprintf( stderr, "..... line search iterations.... %d ( %2.6e, %2.6e)  \n", iterations, *fk1, (*fk + c * alphak * temp) ); 
	return alphak;
}
