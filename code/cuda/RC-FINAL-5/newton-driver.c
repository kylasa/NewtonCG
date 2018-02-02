#include <stdio.h>
#include <stdlib.h>

#include "dataset.h"
#include "sparse_dataset.h"

#include "cuda_environment.h"
#include "newton_cg.h"
#include "utils.h"
#include "cuda_utils.h"
#include "logistic_fn_indicator.h"

#include "softmax_multiclass.h"

cublasHandle_t cublasHandle;
cusparseHandle_t cusparseHandle; 
int BLOCKS, BLOCK_SIZE, BLOCKS_POW_2;
int HESSIAN_SAMPLING_SIZE, GRADIENT_SAMPLING_SIZE; 
void *dscratch;

int main(int argc, char **argv){
	
	// Data variables.
	ForestDataset forestData;
	DeviceDataset devData;
	SCRATCH_AREA	scratch;
	NEWTON_CG_PARAMS params;

	real trainingTime_s, classificationTime_s;
	real trainingTime_t, classificationTime_t;
	int test_case_no = 1;
	int nConIterations;
	int DATASET_TYPE = 1; 

	double l = 1e-6; 
	int max_cg_iterations = -1; 
	double cg_tolerance = 0; 
	int sampling_flag = 0; 
	int gpu = -1; 

	if (argc <= 7) {
		fprintf( stderr, "<exe> dataset lambda .... is the commnad \n"); 
		exit (-1); 
	}

	DATASET_TYPE = atoi( argv[1] ); 
	l = atof ( argv[2] ); 
	max_cg_iterations = atoi (argv[3] ); 
	cg_tolerance = atof( argv[4] ); 
	sampling_flag = atoi( argv[5] );
	gpu = atoi( argv[6] ); 
	HESSIAN_SAMPLING_SIZE = atoi( argv[7] ); 
	GRADIENT_SAMPLING_SIZE = atoi (argv[8] );

        fprintf( stderr, "Dataset: %d, Lambda: %e, CGIterations: %d, CGTolerange: %e, SubSampling: %d, GPU: %d, HSample: %d, GSample: %d \n",
                        DATASET_TYPE, l, max_cg_iterations, cg_tolerance, sampling_flag, gpu, HESSIAN_SAMPLING_SIZE, GRADIENT_SAMPLING_SIZE );


	// Create the CUDA Environment Here. 
	// Memory and device settings here. 
	cuda_env_init (&scratch, gpu);
	#ifdef __debug__
	fprintf( stderr, "Scratch Area initialized ... \n");
	#endif


		switch( DATASET_TYPE ) {
		
                case 1:
                        readMultiDataset (
                                "/home/skylasa/solvers/dataset/raw-data/uci-covertype/train_forest_multi_features.txt",
                                "/home/skylasa/solvers/dataset/raw-data/uci-covertype/train_forest_multi_labels.txt",
                                "/home/skylasa/solvers/dataset/raw-data/uci-covertype/test_forest_multi_features.txt",
                                "/home/skylasa/solvers/dataset/raw-data/uci-covertype/test_forest_multi_labels.txt",
                                &forestData, &scratch, 0, 0 );
                        break;

                case 11:
                        readMultiDataset (
                                "/home/skylasa/solvers/dataset/normalized-data/uci-covertype/train_forest_multi_features.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/uci-covertype/train_forest_multi_labels.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/uci-covertype/test_forest_multi_features.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/uci-covertype/test_forest_multi_labels.txt",
                                &forestData, &scratch, 0, 0 );
                        break;

                case 2:
                        readMultiDataset (
                                "/home/skylasa/solvers/dataset/raw-data/drive-diagnostics/train_mat.txt",
                                "/home/skylasa/solvers/dataset/raw-data/drive-diagnostics/train_vec.txt",
                                "/home/skylasa/solvers/dataset/raw-data/drive-diagnostics/test_mat.txt",
                                "/home/skylasa/solvers/dataset/raw-data/drive-diagnostics/test_vec.txt",
                                &forestData, &scratch, 0, 0 );
                        break;

                case 12:
                        readMultiDataset (
                                "/home/skylasa/solvers/dataset/normalized-data/drive-diagnostics/train_mat.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/drive-diagnostics/train_vec.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/drive-diagnostics/test_mat.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/drive-diagnostics/test_vec.txt",
                                &forestData, &scratch, 0, 0 );
                        break;

                case 3:
                        readMultiDataset (
                                "/home/skylasa/solvers/dataset/raw-data/mnist/train_mat.txt",
                                "/home/skylasa/solvers/dataset/raw-data/mnist/train_vec.txt",
                                "/home/skylasa/solvers/dataset/raw-data/mnist/test_mat.txt",
                                "/home/skylasa/solvers/dataset/raw-data/mnist/test_vec.txt",
                                &forestData, &scratch, 1, 0 );
                        break;

                case 13:
                        readMultiDataset (
                                "/home/skylasa/solvers/dataset/normalized-data/mnist/train_mat.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/mnist/train_vec.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/mnist/test_mat.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/mnist/test_vec.txt",
                                &forestData, &scratch, 1, 0 );
                        break;

                case 4:
                        readCIFARDataset(
                                "/home/skylasa/solvers/dataset/raw-data/cifar-10/cifar-10-batches-bin/",
                                "data_batch_", "test_batch.bin",
                                &forestData, &scratch, 1 );
                        break;

                case 14:
                        readCIFARDataset(
                                "/home/skylasa/solvers/dataset/raw-data/cifar-10/cifar-10-batches-bin/",
                                "data_batch_", "test_batch.bin",
                                &forestData, &scratch, 0 );
                        break;

                case 5:
                        readNewsgroupsDataset (
                                "/home/skylasa/solvers/dataset/raw-data/newsgroups/train_mat.txt",
                                "/home/skylasa/solvers/dataset/raw-data/newsgroups/train_vec.txt",
                                "/home/skylasa/solvers/dataset/raw-data/newsgroups/test_mat.txt",
                                "/home/skylasa/solvers/dataset/raw-data/newsgroups/test_vec.txt",
                                &forestData, &scratch, 0 );
                        break;

                case 15:
                        readNewsgroupsDataset (
                                "/home/skylasa/solvers/dataset/normalized-data/newsgroups/train_mat.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/newsgroups/train_vec.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/newsgroups/test_mat.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/newsgroups/test_vec.txt",
                                &forestData, &scratch, 0 );
                        break;

		//Logistic Datasets Here
                case 6:
                        readMultiDataset (
                                "/home/skylasa/solvers/dataset/raw-data/mushrooms/train_mat.txt",
                                "/home/skylasa/solvers/dataset/raw-data/mushrooms/train_vec.txt",
                                "/home/skylasa/solvers/dataset/raw-data/mushrooms/test_mat.txt",
                                "/home/skylasa/solvers/dataset/raw-data/mushrooms/test_vec.txt",
                                &forestData, &scratch, 1, 0 );
                        break;
                case 16:
                        readMultiDataset (
                                "/home/skylasa/solvers/dataset/normalized-data/mushrooms/train_mat.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/mushrooms/train_vec.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/mushrooms/test_mat.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/mushrooms/test_vec.txt",
                                &forestData, &scratch, 1, 0 );
                        break;

                case 7:
                        readMultiDataset (
                                "/home/skylasa/solvers/dataset/raw-data/ijcnn1/train_mat.txt",
                                "/home/skylasa/solvers/dataset/raw-data/ijcnn1/train_vec.txt",
                                "/home/skylasa/solvers/dataset/raw-data/ijcnn1/test_mat.txt",
                                "/home/skylasa/solvers/dataset/raw-data/ijcnn1/test_vec.txt",
                                &forestData, &scratch, 1, 0 );
                        break;
                case 17:
                        readMultiDataset (
                                "/home/skylasa/solvers/dataset/normalized-data/ijcnn1/train_mat.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/ijcnn1/train_vec.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/ijcnn1/test_mat.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/ijcnn1/test_vec.txt",
                                &forestData, &scratch, 1, 0 );
                        break;
                case 8:
                        readMultiDataset (
                                "/home/skylasa/solvers/dataset/raw-data/gisette/gisette_train.data",
                                "/home/skylasa/solvers/dataset/raw-data/gisette/gisette_train.labels01",
                                "/home/skylasa/solvers/dataset/raw-data/gisette/gisette_valid.data",
                                "/home/skylasa/solvers/dataset/raw-data/gisette/gisette_valid.labels01",
                                &forestData, &scratch, 1, 0 );
                        break;
                case 18:
                        readMultiDataset (
                                "/home/skylasa/solvers/dataset/normalized-data/gisette/gisette_train.data",
                                "/home/skylasa/solvers/dataset/normalized-data/gisette/gisette_train.labels01",
                                "/home/skylasa/solvers/dataset/normalized-data/gisette/gisette_valid.data",
                                "/home/skylasa/solvers/dataset/normalized-data/gisette/gisette_valid.labels01",
                                &forestData, &scratch, 1, 0 );
                        break;
                case 9:
                        readNewsgroupsDataset (
                                "/home/skylasa/solvers/dataset/raw-data/rcv1/train_mat.txt",
                                "/home/skylasa/solvers/dataset/raw-data/rcv1/train_vec.txt",
                                "/home/skylasa/solvers/dataset/raw-data/rcv1/test_mat.txt",
                                "/home/skylasa/solvers/dataset/raw-data/rcv1/test_vec.txt",
                                &forestData, &scratch, 1 );
                        break;

		//Sparse Logistic Datasets Here
                case 10:
                        readNewsgroupsDataset (
                                "/home/skylasa/solvers/dataset/raw-data/real-sim/train_mat.txt",
                                "/home/skylasa/solvers/dataset/raw-data/real-sim/train_vec.txt",
                                "/home/skylasa/solvers/dataset/raw-data/real-sim/test_mat.txt",
                                "/home/skylasa/solvers/dataset/raw-data/real-sim/test_vec.txt",
                                &forestData, &scratch, 1 );
                        break;
                case 20:
                        readNewsgroupsDataset (
                                "/home/skylasa/solvers/dataset/normalized-data/real-sim/train_mat.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/real-sim/train_vec.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/real-sim/test_mat.txt",
                                "/home/skylasa/solvers/dataset/normalized-data/real-sim/test_vec.txt",
                                &forestData, &scratch, 1 );
                        break;
		}

	#ifdef __debug__
		fprintf( stderr, "Done with initialization of the dataset .... \n");
		fprintf( stderr, "Blocks for %d data points... \n", forestData.rows);
	#endif

        	compute_blocks (&BLOCKS, &BLOCK_SIZE, forestData.trainSize);
        	compute_nearest_pow_2 (BLOCKS, &BLOCKS_POW_2);
		if (BLOCKS_POW_2 < 32) BLOCKS_POW_2 = 32;
	#ifdef __debug__
		fprintf ( stderr, "Blocks: %d, BlockSize: %d, Power_2: %d\n", BLOCKS, BLOCK_SIZE, BLOCKS_POW_2);
	#endif


		// Move the data to the Device. 
		if ((forestData.trainSet == NULL) && (forestData.testSet == NULL))
		{	
			initialize_device_data_sparse( &forestData, &devData );
			initMatDescriptors ( &devData ); 
			convertToCSR ( &devData, scratch.devWorkspace ); 

			initMatDescriptorsForSampling( &devData ); 
			initMatDescriptorsForSparseSampling( &devData ); 
		} else {
			initialize_device_data( &forestData, &devData );
			initMatDescriptorsForSampling( &devData ); 
		}
		
	#ifdef __debug__
		fprintf( stderr, "Inittialized the Device with the dataset ... \n");
	#endif

		//Train the dataset here. 
		params.max_iterations = 100;
		params.tolerance = 1e-5;
		params.iflag = 0;

		params.lambda = l;
		params.max_cg_iterations = max_cg_iterations; 
		params.cg_tolerance = cg_tolerance;

		if (GRADIENT_SAMPLING_SIZE == 100)
			params.gx_sampling = 0; 
		else
			params.gx_sampling = sampling_flag; 
		params.hx_sampling = sampling_flag;

fprintf( stderr, " Gradient Sample: %d, Hessian Sample: %d \n", devData.gradientSampleSize, devData.hessianSampleSize ); 
		if (sampling_flag == 0) {

			devData.gradientSampleSize = 0; 
			devData.hessianSampleSize = 0; 
		}

		fprintf( stderr, "Start of TestCase: %d\n", 	test_case_no);
		trainingTime_s = Get_Time ();
		/*
		if (forestData.numclasses == 1) 
			nConIterations = newton_cg( &forestData, &devData, &params, &scratch );
		else 
		*/
			nConIterations = newton_cg_multi_optimized( &forestData, &devData, &params, &scratch);
		trainingTime_t = Get_Timing_Info( trainingTime_s );	
	#ifdef __debug__
		fprintf( stderr, "Done with training .... \n");
	#endif
	
		//exit (-1); 

		//Predict the testing set here. 
		real accuracy = 0;
		classificationTime_s = Get_Time ();
		/*
		if (forestData.numclasses == 1) {
			logistic_regression_predict( devData.testSet, &devData.spTest, devData.weights, devData.testLabels, 
					forestData.testLabels, forestData.testSize, forestData.cols, 
					&accuracy, scratch.devWorkspace, scratch.hostWorkspace );
		} else {
		*/
			accuracy = softmax_predict(&devData.spTest, devData.testSet, forestData.testLabels, 
					devData.weights, devData.testSize, devData.cols, devData.numclasses, 
					scratch.hostWorkspace, scratch.devWorkspace, 1, forestData.testSet);
		//}
		classificationTime_t = Get_Timing_Info( classificationTime_s );
		//fprintf( stderr, "Start of TestCase: %d\n", 	test_case_no);
		fprintf( stderr, "Dataset: %d \n", 		DATASET_TYPE ); 
		fprintf( stderr, "NumClasses: %d\n", 		devData.numclasses ); 
		fprintf( stderr, "Lambda: %e\n", 		params.lambda );
		fprintf( stderr, "NewtonIterations: %d\n", 	params.max_iterations);
		fprintf( stderr, "NewtonTolerance: %e\n", 	params.tolerance);
		fprintf( stderr, "CGIterations: %d\n", 		params.max_cg_iterations);
		fprintf( stderr, "CGTolerance: %e\n", 		params.cg_tolerance );
		fprintf( stderr, "DataSetSize: %d\n", 		forestData.rows );
		fprintf( stderr, "TrainingSize: %d\n", 		forestData.trainSize);
		fprintf( stderr, "Features: %d\n", 		forestData.cols );
		fprintf( stderr, "TrainingTime: %d\n", 		(unsigned int)(trainingTime_t * 1000)  );
		fprintf( stderr, "TestingSize: %d\n", 		forestData.testSize );
		fprintf( stderr, "ClassificationTime: %d\n", 	(unsigned int)(classificationTime_t*1000)  );
		fprintf( stderr, "TestAccuracy:  %3.2f\n", 	accuracy );
		fprintf( stderr, "NewtonIterationsCon: %d\n", 	nConIterations );
		fprintf( stderr, "NewtonConvergence: %d\n", 	(int)params.iflag );
		fprintf( stderr, "End of TestCase: %d\n", 	test_case_no);
		fprintf( stderr, "\n\n\n");

		//cleanup the dataset pointers here. 
		cleanup_dataset(&forestData, &devData );

		test_case_no ++;

	//Cleanup host/device Here. 
	cuda_env_cleanup(&scratch);

	return 0;
}
