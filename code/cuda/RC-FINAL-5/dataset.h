#ifndef _H_DATASET__
#define _H_DATASET__

#include <cuda_types.h>

typedef struct dataset{ 
	real *trainSet;	
	real *trainLabels;
	real *testSet;
	real *testLabels; 
	int trainSize; 
	int testSize; 

	int rows; 
	int cols;
	int numclasses; 

	int *trainRowPtr, *trainColPtr, *testRowPtr, *testColPtr; 
	real *trainValPtr, *testValPtr; 
	int trainNNZ, testNNZ; 
} ForestDataset;

typedef struct spData {
	int *rowPtr, *colPtr, *rowCsrPtr; 
	real *valPtr; 

	int nnz; 

	//int *cscRowPtr, *cscColPtr; 
	//real *cscValPtr; 

	real *sortedVals;
	int *P; 

	cusparseMatDescr_t descr; 	

} SparseDataset; 

typedef struct devDataSet{
	real *trainSet; 
	real *trainLabels;
	real *testSet; 
	real *testLabels; 

	real *weights; 
	int rows; 
	int cols;

	int testSize;

	int numclasses; 

	SparseDataset spTrain; 
	SparseDataset spTest; 

	//subsampling part here. 
	real *sampledGradientTrainSet; 
	real *sampledGradientTrainLabels; 
	int gradientSampleSize; 
	SparseDataset spGradientSample; 

	real *sampledHessianTrainSet; 
	int hessianSampleSize; 
	SparseDataset spHessianSample; 

	SparseDataset spSampledGradientTrain; 
	SparseDataset spSampledHessianTrain; 
	
}DeviceDataset;

typedef struct params{ 
	real *sigma; 
	real *mu; 
}GAUSSIAN_PARAMS;

void printDataset( ForestDataset *t );


void readMultiDataset( char *f_train_features, char *f_train_labels,
                char *f_test_features, char *f_test_labels, ForestDataset *data, SCRATCH_AREA *s, int offset, int bias);
void readCIFARDataset( char *dir, char *train, char *test, ForestDataset *data, SCRATCH_AREA *s, int);
void readNewsgroupsDataset( char *train_features, char *train_labels,
                                char *test_features, char *test_labels,
                                ForestDataset *data, SCRATCH_AREA *s, int offset);
void readBinaryMatFile( char *train_features, char *train_labels,
                                char *test_features, char *test_labels,
                                ForestDataset *data, SCRATCH_AREA *s, int offset);

//int tokenize_learn_multiclass( char *line, real* t, int curIndex, int *counters, int **idx);
int tokenize_string( char *line, real *out, int bias );
void tokenize_populate(char *line, real *train_set, int *count, int size, int bias);

int tokenize_binary_string( char *line, int bias, int *nnz);
int tokenize_binary_populate( char *line, int bias, int *row, int *col, real *val, real *label, int rowNum );


void initialize_device_data( ForestDataset *s, DeviceDataset *t);
void initialize_device_data_sparse( ForestDataset *s, DeviceDataset *t );
void cleanup_dataset( ForestDataset *s, DeviceDataset *t);

real findMaxInDataset( real *src, int rows, int cols );
void preprocessDataset( real *src, int rows, int cols, real maxval);



#endif
