#include <newton_cg.h>
#include "logistic_fn_indicator.h"

#include "cuda_utils.h"
#include "conjugate_gradient.h"
#include "linesearch.h"

#include "print_utils.h"
#include "logistic_fn_indicator.h"
#include "utils.h"

#include "softmax_multiclass.h"
#include "subsampling_helpers.h"
#include "sparse_dataset.h"

#define ALLOTED_TIME (120 * 60)

int newton_cg( ForestDataset *host, DeviceDataset *data, NEWTON_CG_PARAMS *params, SCRATCH_AREA *scratch){

	int iterations, cg_iterations; 
	real snorm, gxnorm, rel_residual, best_rel_residual; 
	real alpha, alphak;

	real train_accuracy, test_accuracy;
	real iteration_start, iteration_total, simulation_total;

	//device
	real *devPtr = (real *)scratch->devWorkspace;
	real *xx = devPtr;
	real *s = xx + data->cols;
	real *s_best = s + data->cols;
	real *gradient = s_best + data->cols;
	//real *hessian = gradient+ data->cols;
	//real *nextDevPtr = hessian + (data->cols * data->cols);
	real *nextDevPtr = gradient + data->cols; 

	real *nextHostPtr = (real *)scratch->hostWorkspace;

	//pageLock
	real *train_function, *test_function; 
	train_function = scratch->pageLckWorkspace; 
	test_function = & (scratch->pageLckWorkspace[1] );

	//Subsampling here. 
	//extract the subsampled gradient here. 
fprintf( stderr, "Running the Logistic Regression..... solver, %d, %d, %d, %d \n", data->rows, data->cols, data->numclasses, data->testSize); 

	//1.  get the hessian and gradient. 
	if (params->gx_sampling >= 1) {

		data->gradientSampleSize = (GRADIENT_SAMPLING_SIZE * data->rows) / 100; 

		if (data->trainSet != NULL && data->testSet != NULL) {
                	prepareForSampling( &data->spGradientSample, data->sampledGradientTrainLabels, data->trainLabels, 
					data->rows, data->gradientSampleSize, (int *)nextHostPtr);
                	convertGradientSampleToCSR( &data->spGradientSample, data->gradientSampleSize, data->cols, nextDevPtr );
                	sampleDataset(&data->spGradientSample, data->trainSet, data->rows, data->cols, 
				data->numclasses, data->sampledGradientTrainSet, data->gradientSampleSize);
		} else {
			//handle sparse datasets here. 
                	prepareForSampling( &data->spGradientSample, data->sampledGradientTrainLabels, data->trainLabels, 
					data->rows, data->gradientSampleSize, (int *)nextHostPtr);
                	convertGradientSampleToCSR( &data->spGradientSample, data->gradientSampleSize, data->cols, nextDevPtr );

			sampleSparseDataset( &data->spGradientSample, &data->spTrain, 
					data->rows, data->cols, data->numclasses, 
					&data->spSampledGradientTrain, data->gradientSampleSize ); 
			fprintf( stderr, "Done extracting the sparse dataset ..... \n"); 
		}
		logistic_fn_indicator_gx( data->sampledGradientTrainSet, &data->spSampledGradientTrain, data->sampledGradientTrainLabels, 
						data->weights, params->lambda, data->gradientSampleSize, data->cols, gradient, 
						nextDevPtr, nextHostPtr, params->gx_sampling, data->rows ); 
					
	} else {
		logistic_fn_indicator_gx( data->trainSet, &data->spTrain, data->trainLabels, data->weights, params->lambda, 
						data->rows, data->cols, gradient, nextDevPtr, nextHostPtr, params->gx_sampling, data->rows);
	}

	//norm of gradient. 
	cublasCheckError( cublasDnrm2( cublasHandle, data->cols, gradient, 1, &gxnorm ));

	iterations = 0;
	snorm = 100;
	//gxnorm = 100;

	rel_residual = 0; 
	best_rel_residual = 0; 
	train_accuracy = 0; 
	*train_function = 0; 
	test_accuracy = 0; 
	*test_function = 0; 
	iteration_total = 0; 
	simulation_total = 0;

#ifdef __debug__
	fprintf( stderr, "iteration \t norm(gradient) \t Rel_Residual \t CG-ITERATIONS \t Train_Accu \t Obj_Val_Train \t Test_Accu \t Obj_Val_Test \n");

        logistic_regression_predict( data->trainSet, &data->spTrain, data->weights, data->trainLabels,
                                        host->trainLabels, host->trainSize, host->cols,
                                        &train_accuracy, nextDevPtr, nextHostPtr );
        logistic_regression_predict( data->testSet, &data->spTest, data->weights, data->testLabels,
                                        host->testLabels, host->testSize, host->cols,
                                        &test_accuracy, nextDevPtr, nextHostPtr );

	logistic_fn_indicator( data->trainSet, &data->spTrain, data->trainLabels, data->weights, params->lambda, data->rows, data->cols, train_function, nextDevPtr, nextHostPtr);
	logistic_fn_indicator( data->testSet, &data->spTest, data->testLabels, data->weights, params->lambda, data->testSize, data->cols, test_function, nextDevPtr, nextHostPtr);

	fprintf( stderr, "%9d \t %e \t %e \t %d \t %3.2f \t %e \t %3.2f \t %e \t %d\n", 
			iterations, gxnorm, rel_residual, 0, train_accuracy, *train_function, 
			test_accuracy, *test_function, (unsigned int)(iteration_total * 1000) );
#endif

	while (iterations < params->max_iterations){

		iteration_start = Get_Time( );

		//alpha = -1.;
		//cublasCheckError ( cublasDscal( cublasHandle, data->cols, &alpha, gradient, 1) );
		
		//conjugate gradient to solve Hx = gradient here. 
		cuda_memset( s_best, 0, data->cols, ERROR_MEM_SET );
		cuda_memset( s, 0, data->cols, ERROR_MEM_SET );
		cg_iterations = Cublas_CG_Logistic( data, params, gradient, s, s_best, &rel_residual, 
						nextDevPtr, nextHostPtr, scratch->pageLckWorkspace ); 

		alphak = cg_linesearch( s_best, data->weights, 0.5, 1e-6, &data->spTrain, 
				(real *)data->trainSet, (real *)data->trainLabels, 
				params->lambda, data->rows, data->cols, data->numclasses, 
				gradient, xx, nextDevPtr, nextHostPtr, (real *)scratch->pageLckWorkspace);

//fprintf( stderr, "alphaK --> %e \n", alphak ); 
		
		alpha = alphak;
		cublasCheckError( cublasDaxpy( cublasHandle, data->cols, &alpha, s_best, 1, data->weights, 1) );

		if (params->gx_sampling >= 1) {
			if (data->trainSet != NULL && data->testSet != NULL) {
                		prepareForSampling( &data->spGradientSample, data->sampledGradientTrainLabels, data->trainLabels, 
					data->rows, data->gradientSampleSize, (int *)nextHostPtr);
                			convertGradientSampleToCSR( &data->spGradientSample, data->gradientSampleSize, data->cols, nextDevPtr );
                		sampleDataset(&data->spGradientSample, data->trainSet, data->rows, data->cols, 
					data->numclasses, data->sampledGradientTrainSet, data->gradientSampleSize);
			} else {
				//handle sparse datasets here. 
                		prepareForSampling( &data->spGradientSample, data->sampledGradientTrainLabels, data->trainLabels, 
					data->rows, data->gradientSampleSize, (int *)nextHostPtr);
                		convertGradientSampleToCSR( &data->spGradientSample, data->gradientSampleSize, data->cols, nextDevPtr );

				sampleSparseDataset( &data->spGradientSample, &data->spTrain, 
					data->rows, data->cols, data->numclasses, 
					&data->spSampledGradientTrain, data->gradientSampleSize ); 
			}
			logistic_fn_indicator_gx( data->sampledGradientTrainSet, &data->spSampledGradientTrain, data->sampledGradientTrainLabels, 
						data->weights, params->lambda, data->gradientSampleSize, data->cols, gradient, 
						nextDevPtr, nextHostPtr, params->gx_sampling, data->rows ); 
					
		} else {
			logistic_fn_indicator_gx( data->trainSet, &data->spTrain, data->trainLabels, data->weights, params->lambda, 
						data->rows, data->cols, gradient, nextDevPtr, nextHostPtr, params->gx_sampling, data->rows);
		}

		cublasCheckError( cublasDnrm2( cublasHandle, data->cols, gradient, 1, &gxnorm ));
		cublasCheckError( cublasDnrm2( cublasHandle, data->cols, s_best, 1, &snorm));

#ifdef __debug__

		iteration_total = Get_Timing_Info( iteration_start );
		simulation_total += iteration_total;

                logistic_regression_predict( data->trainSet, &data->spTrain, data->weights, data->trainLabels,
                                        host->trainLabels, host->trainSize, host->cols,
                                        &train_accuracy, nextDevPtr, nextHostPtr );
                logistic_regression_predict( data->testSet, &data->spTest, data->weights, data->testLabels,
                                        host->testLabels, host->testSize, host->cols,
                                        &test_accuracy, nextDevPtr, nextHostPtr );

		logistic_fn_indicator( data->trainSet, &data->spTrain, data->trainLabels, data->weights, params->lambda, 
					data->rows, data->cols, train_function, nextDevPtr, nextHostPtr);
		logistic_fn_indicator( data->testSet, &data->spTest, data->testLabels, data->weights, params->lambda, 
					data->testSize, data->cols, test_function, nextDevPtr, nextHostPtr);

		fprintf( stderr, "%9d \t %e \t %e \t %d \t %3.2f \t %e \t %3.2f \t %e \t %d\n", 
			iterations+1, gxnorm, rel_residual, cg_iterations, train_accuracy, *train_function, 
			test_accuracy, *test_function, (unsigned int)(iteration_total * 1000) );
	
#endif

		iterations ++; 
		if (gxnorm <= params->tolerance) break;

		if (((unsigned int)(simulation_total)) >=  ALLOTED_TIME ) {
			fprintf( stderr, "Exceeded the Time limitation for the simulation..... %d, %d \n", ((unsigned int)(simulation_total )), ALLOTED_TIME ); 
			break;
		}
	}

	if (gxnorm >= params->tolerance)
		params->iflag = 1;

	return iterations;
}



int newton_cg_multi_optimized( ForestDataset *host, DeviceDataset *data, NEWTON_CG_PARAMS *params, SCRATCH_AREA *scratch){

	int iterations, cg_iterations; 
	real snorm, gxnorm, rel_residual, temp; 
	real alpha, alphak;

	real best_rel_residual; 

#ifdef __STATISTICS__
	//statistics here. 
	real train_accuracy, train_function; 
	real test_accuracy, test_function; 
	real iteration_start, iteration_total, simulation_total;
#endif
	
	int classes_to_solve = data->numclasses;

	//device
	real *xx = (real *)scratch->devWorkspace;
	real *s = xx + data->cols * classes_to_solve; 
	real *s_best = s + data->cols * classes_to_solve; 

	//auxiliary storeage 
	real *gradient = s_best + data->cols * classes_to_solve; 
	real *Hv = gradient + data->cols * classes_to_solve; 
	real *HXW = Hv + classes_to_solve * data->cols; 
	//real *expSumVec = XW + rows * classes_to_solve; 

	//scratch area 
	real *nextDevPtr = HXW + data->rows* classes_to_solve; 
	real *nextHostPtr = (real *)scratch->hostWorkspace;
	real *nextPageLckPtr = (real *) scratch->pageLckWorkspace; 

	real s_norm, s_best_norm; 


	//1.  get the hessian and gradient. 
	if (params->hx_sampling >= 1) 
		data->hessianSampleSize = (HESSIAN_SAMPLING_SIZE * data->rows)/100; 

	if (params->gx_sampling >= 1) {

		data->gradientSampleSize = (GRADIENT_SAMPLING_SIZE * data->rows) / 100; 
		data->spGradientSample.nnz = data->gradientSampleSize; 

		
		if (data->trainSet != NULL && data->testSet != NULL) {
                	prepareForSampling( &data->spGradientSample, data->sampledGradientTrainLabels, data->trainLabels, 
					data->rows, data->gradientSampleSize, (int *)nextHostPtr);
                	convertGradientSampleToCSR( &data->spGradientSample, data->gradientSampleSize, data->cols, nextDevPtr );
                	sampleDataset(&data->spGradientSample, data->trainSet, data->rows, data->cols, 
				classes_to_solve, data->sampledGradientTrainSet, data->gradientSampleSize);
		} else {
			//handle sparse datasets here. 
                	prepareForSampling( &data->spGradientSample, data->sampledGradientTrainLabels, data->trainLabels, 
					data->rows, data->gradientSampleSize, (int *)nextHostPtr);
                	convertGradientSampleToCSR( &data->spGradientSample, data->gradientSampleSize, data->cols, nextDevPtr );

			sampleSparseDataset( &data->spGradientSample, &data->spTrain, 
					data->rows, data->cols, classes_to_solve, 
					&data->spSampledGradientTrain, data->gradientSampleSize ); 
		}

		softmax_multiclass_gx_subsampled(&data->spTrain, data->trainSet, data->trainLabels, data->rows, data->cols,
                               classes_to_solve, data->weights, params->lambda,
				gradient, nextDevPtr, nextHostPtr, scratch->pageLckWorkspace, 
				&data->spGradientSample, data->sampledGradientTrainSet, &data->spSampledGradientTrain, 
				data->sampledGradientTrainLabels, data->gradientSampleSize, params->gx_sampling);
		printVector( gradient, 10, NULL ); 
		
	} else {
		computeHXW(&data->spTrain, data->trainSet, data->rows, data->cols, classes_to_solve, data->weights, HXW, 0 ); 

		softmax_multiclass_gx_optimized(&data->spTrain, data->trainSet, data->trainLabels, data->rows, data->cols,
                               classes_to_solve, data->weights, params->lambda, HXW, 
				gradient, nextDevPtr, nextHostPtr, scratch->pageLckWorkspace);
	}
	//printVector( gradient, 20, NULL ); 
	/*
	softmax_multiclass_gx(data->trainSet, data->trainLabels, data->rows, data->cols,
                                classes_to_solve, data->weights, params->lambda,
				gradient, nextDevPtr, nextHostPtr, scratch->pageLckWorkspace);
	*/

	//2. Initialization Here. 
	iterations = 0;
	snorm = 100;
	gxnorm = 100;
	rel_residual = 100; 

	cublasCheckError( cublasDnrm2( cublasHandle, classes_to_solve * data->cols, gradient, 1, &gxnorm ));

#ifdef __STATISTICS__
	iteration_total = 0; 
	simulation_total = 0; 

	test_function = softmax_multiclass_fx (&data->spTest, data->testSet, data->testLabels, data->testSize, data->cols, 
		classes_to_solve, data->weights, params->lambda,
		nextDevPtr, nextHostPtr, scratch->pageLckWorkspace ); 
	train_function = softmax_multiclass_fx (&data->spTrain, data->trainSet, data->trainLabels, host->trainSize, data->cols, 
			classes_to_solve, data->weights, params->lambda,
			nextDevPtr, nextHostPtr, scratch->pageLckWorkspace ); 


        test_accuracy = softmax_predict(&data->spTest, data->testSet, host->testLabels, data->weights, data->testSize,
                                data->cols, classes_to_solve, nextHostPtr, nextDevPtr, 1, NULL);
	train_accuracy = softmax_predict( &data->spTrain, data->trainSet, host->trainLabels, data->weights, host->trainSize, 
				data->cols, classes_to_solve, nextHostPtr, nextDevPtr, 1, NULL ); 


	fprintf( stderr, "iteration \t norm(gradient) \t Rel_Residual \t CG-ITERATIONS \t Train_Accu \t Obj_Val_Train \t Test_Accu \t Obj_Val_Test \n");
	fprintf( stderr, "%9d \t %e \t %e \t %d \t %3.2f \t %e \t %3.2f \t %e \t %d\n", 
			iterations, gxnorm, rel_residual, 0, train_accuracy, train_function, 
			test_accuracy, test_function, (unsigned int)(iteration_total * 1000) );

#endif

	while (iterations < params->max_iterations){

#ifdef __STATISTICS__
		//statistics Here. 
		iteration_start = Get_Time( );
#endif
		//negative gradient
		alpha = -1.;
		cublasCheckError ( cublasDscal( cublasHandle, classes_to_solve * data->cols, &alpha, gradient, 1) );

		cuda_memset( s, 0, classes_to_solve * data->cols * sizeof(real), ERROR_MEM_SET );
		cuda_memset( s_best, 0, classes_to_solve * data->cols * sizeof(real), ERROR_MEM_SET );

		cg_iterations = Cublas_CG_multi_optimized( &data->spTrain, data->trainSet, gradient, data->weights, s, s_best, params->lambda, 
					data->rows, data->cols, classes_to_solve, HXW, 
					nextDevPtr, nextHostPtr, scratch->pageLckWorkspace, 
					params->max_cg_iterations, params->cg_tolerance, &rel_residual, &best_rel_residual, 
					&data->spHessianSample, data->sampledHessianTrainSet, 
					&data->spSampledHessianTrain, data->hessianSampleSize, params->hx_sampling); 

		//compute the relative residual here. 
		// || H*x - g || / || g ||
		cublasCheckError( cublasDnrm2( cublasHandle, classes_to_solve * data->cols, gradient, 1, &gxnorm ));

		//change gradient back
		alpha = -1.;
		cublasCheckError ( cublasDscal( cublasHandle, classes_to_solve * data->cols, &alpha, gradient, 1) );
		alphak = cg_linesearch( s_best, data->weights, 0.5, 1e-6, &data->spTrain, (real *)data->trainSet, (real *)data->trainLabels, 
					params->lambda, data->rows, data->cols, classes_to_solve, gradient, xx, 
					nextDevPtr, nextHostPtr, (real *)scratch->pageLckWorkspace);

		alpha = alphak;
		cublasCheckError( cublasDaxpy( cublasHandle, classes_to_solve * data->cols, &alpha, s_best, 1, data->weights, 1) );

		

		if (params->gx_sampling >= 1) {

			if (data->trainSet != NULL && data->testSet != NULL) {
                		prepareForSampling( &data->spGradientSample, data->sampledGradientTrainLabels, data->trainLabels, 
						data->rows, data->gradientSampleSize, (int *)nextHostPtr);
                		convertGradientSampleToCSR( &data->spGradientSample, data->gradientSampleSize, data->cols, nextDevPtr );
                		sampleDataset(&data->spGradientSample, data->trainSet, data->rows, data->cols, 
					classes_to_solve, data->sampledGradientTrainSet, data->gradientSampleSize);
			} else {
				//handle sparse datasets here. 
                		prepareForSampling( &data->spGradientSample, data->sampledGradientTrainLabels, data->trainLabels, 
					data->rows, data->gradientSampleSize, (int *)nextHostPtr);
                		convertGradientSampleToCSR( &data->spGradientSample, data->gradientSampleSize, data->cols, nextDevPtr );

				sampleSparseDataset( &data->spGradientSample, &data->spTrain, 
					data->rows, data->cols, classes_to_solve, 
					&data->spSampledGradientTrain, data->gradientSampleSize ); 
			}

			softmax_multiclass_gx_subsampled(&data->spTrain, data->trainSet, data->trainLabels, data->rows, data->cols,
                               classes_to_solve, data->weights, params->lambda,
				gradient, nextDevPtr, nextHostPtr, scratch->pageLckWorkspace, 
				&data->spGradientSample, data->sampledGradientTrainSet, &data->spSampledGradientTrain, 
				data->sampledGradientTrainLabels, data->gradientSampleSize, params->gx_sampling);
	
		} else {
			//update here. 
			computeHXW( &data->spTrain, data->trainSet, data->rows, data->cols, classes_to_solve, data->weights, HXW, 0 ); 

			softmax_multiclass_gx_optimized(&data->spTrain, data->trainSet, data->trainLabels, data->rows, data->cols,
                                classes_to_solve, data->weights, params->lambda, HXW, 
				gradient, nextDevPtr, nextHostPtr, scratch->pageLckWorkspace);
		}
		
#ifdef __STATISTICS__
		iteration_total = Get_Timing_Info( iteration_start );
		simulation_total += iteration_total;
		//fprintf( stderr, "Total time per iteration ---- > %f \n", iteration_total ); 

		//per iteration statistics here. 
                test_accuracy = softmax_predict(&data->spTest, data->testSet, host->testLabels, data->weights, data->testSize,
                                data->cols, classes_to_solve, nextHostPtr, nextDevPtr, 1, NULL);
		train_accuracy = softmax_predict( &data->spTrain, data->trainSet, host->trainLabels, data->weights, host->trainSize, 
				data->cols, classes_to_solve, nextHostPtr, nextDevPtr, 1, NULL ); 
		test_function = softmax_multiclass_fx(&data->spTest, data->testSet, data->testLabels, data->testSize, data->cols, 
			classes_to_solve, data->weights, params->lambda, nextDevPtr, nextHostPtr, scratch->pageLckWorkspace ); 
		train_function = softmax_multiclass_fx(&data->spTrain, data->trainSet, data->trainLabels, data->rows, data->cols, 
			classes_to_solve, data->weights, params->lambda, nextDevPtr, nextHostPtr, scratch->pageLckWorkspace ); 

		fprintf( stderr, "%9d \t %e \t %e \t %d \t %3.2f \t %e \t %3.2f \t %e \t %d\n", 
			iterations+1, gxnorm, rel_residual, cg_iterations, 
			train_accuracy, train_function, test_accuracy, test_function, (unsigned int)(iteration_total * 1000) );
#endif

		iterations ++; 
		if (gxnorm <= params->tolerance) break;

		if (((unsigned int)(simulation_total )) >=  ALLOTED_TIME ) {
			fprintf( stderr, "Exceeded the Time limitation for the simulation..... %d, %d \n", ((unsigned int)(simulation_total )), ALLOTED_TIME ); 
			break;
		}
	}

	if (gxnorm >= params->tolerance)
		params->iflag = 1;

	return iterations;
}
