#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <dataset.h>
#include <cuda_utils.h>
#include <utils.h>

#include <print_utils.h>

#define SAMPLING_BUFFER_EXTENSION	10

#define MAX_LINE 	256 * 1024
#define MAX_IDX 	256 * 1024

#define HEAP_LINE_SIZE 4 * 1024 * 1024

#define CIFAR_LINE_SIZE 	3073

void swap (real *a, real *b, real *t){
	*t = *a;
	*a = *b;
	*b = *t;
}

real findMaxInDataset( real *src, int rows, int cols )
{
	int maxval = 0; 
	for (int i = 0; i < rows * cols; i ++)
		if (maxval < src[ i ]) maxval = src[i]; 
	return maxval; 
}

void preprocessDataset( real *src, int rows, int cols, real maxval)
{
	if (maxval > 0){ 
		for (int i = 0; i < rows * cols; i ++)
			//src[i] = maxval - src[i];
			src[i] = src[i] - maxval;
	}
}

void convertToColumnMajor (real *src, int rows, int cols, real *tgt ) {
	for (int i = 0; i < rows; i ++ )
		for (int j = 0; j < cols; j ++)
			tgt[j * rows + i]  = src[i * cols + j];
}

void convertRowStochastic( real *src, int rows, int cols ) {
	real sum = 0; 
	for (int i = 0; i < rows; i ++ ) {
		for (int j = 0; j < cols; j ++)
			sum += src[ i * cols + j ];
		for (int j = 0; j < cols; j ++)
			src[ i * cols + j ] = src[ i * cols + j] / sum;
	}
}

void convertColumnStochastic( real *src, int rows, int cols ){
	real maxval = 0; 
	for (int c = 0; c < cols; c ++){
		maxval = src[ c * rows ]; 
		for (int r = 1; r < rows; r ++){
			if (maxval < src[ c * rows + r ])
				maxval = src[ c * rows + r ];	
		}

		if (maxval > 1) {
			for (int r = 1; r < rows; r ++){
				src[ c * rows + r] /= maxval;
			}
		}
		//fprintf( stderr, " Done with Column: %d, maxval: %f \n", c, maxval );
	}
}

void columnNormalize( real *src, int rows, int cols, real *train, int tr ){
        real norm = 0;
        for (int c = 0; c < cols; c ++){
                norm = pow( src[ c * rows ], 2. );
                for (int r = 1; r < rows; r ++) {
                        norm += pow( src[ c * rows + r ], 2. );
                }
                for (int r = 0; r < tr; r ++){
                        norm += pow( src[ c * tr + r ], 2. );
                }

                if (norm < 1e-8) {
                        norm = sqrt( norm );
                        for (int r = 0; r < rows; r ++)
                                src[ c * rows + r ] /= norm;

                        for (int r = 0; r < tr; r ++)
                                train[ c * tr + r ] /= norm;
                }
        }
}

real computeMaxValue (real *labels, int count ) {
	real maxval = 0; 
	for (int i = 0; i < count; i ++ ) 
		if (maxval < labels[i] )
			maxval = labels[i]; 

	return maxval; 
}

void writeDataset( real *features, real *labels, int rows, int cols, char *filename, char *vectorname)
{
	FILE *dataset_file;

	if ( (dataset_file = fopen(filename, "w")) == NULL ) { 
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}
	
	for (int i = 0; i < rows; i ++){
		fprintf (dataset_file, "%4.6f", features[ i * cols ] );
		for (int j = 1; j < cols; j ++){
			fprintf( dataset_file, ",%4.6f", features[ i * cols + j ] );
		}
		fprintf( dataset_file, "\n");
	}
	fclose (dataset_file);

	if ( (dataset_file = fopen(vectorname, "w")) == NULL ) { 
		fprintf( stderr, "Error opening the labels.... !\n" );
		exit( -1 );
	}
	
	for (int i = 0; i < rows; i ++){
		fprintf (dataset_file, "%d\n", (int)labels[ i ] );
	}
	fclose (dataset_file);
}

void readBinaryMatFile( char *f_train_features, char *f_train_labels, 
				char *f_test_features, char *f_test_labels, 
				ForestDataset *data, SCRATCH_AREA *s, int offset)
{
	FILE *dataset_file; 
	char line[MAX_LINE]; 
	int numLines = 0;
	int NUM_CLASSES = 20;
	size_t output; 
	int idx = 0; 
	int i;
	real cols[3]; 
	int max_train_col, max_test_col; 
	int max_train_row, max_test_row; 

	char filename[MAX_LINE]; 

	real *scratch = s->hostWorkspace;

	int *train_row_id, *train_col_id; 
	int *test_row_id, *test_col_id; 

	real *train_val, *train_vec; 
	real *test_val, *test_vec; 
	
	int train_nnz, test_nnz; 

	int cur_column; 
	int *rowPtr, *colPtr; 
	real *valPtr, *labelPtr; 
	int rowNNZ; 
	int minCol = 1000000; 

	char *heapLine = (char *)malloc (HEAP_LINE_SIZE); 
	
	if ( (dataset_file = fopen(f_train_features, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}

	max_train_row = max_train_col = 0; 
	train_nnz = 0;
	while (!feof( dataset_file) ){
		memset( heapLine, 0, HEAP_LINE_SIZE);

		fgets( heapLine, HEAP_LINE_SIZE, dataset_file);
		if (heapLine[0] == 0) break;

		cur_column = tokenize_binary_string( heapLine, 0, &train_nnz); 

		if (max_train_col < cur_column) max_train_col = cur_column; 
		if (minCol > cur_column) minCol = cur_column; 
		
		numLines ++;
	}
	max_train_row = numLines; 

	fclose( dataset_file ); 
	fprintf( stderr, "Done with reading %d points from the input files ....(%d, %d), NNZ: %d, %d\n", 
			numLines, max_train_row, max_train_col, train_nnz, minCol ); 

	if ( (dataset_file = fopen(f_test_features, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}

	max_test_row = max_test_col = numLines = 0; 
	test_nnz = 0; 
	minCol = 10000000; 
	while (!feof( dataset_file) ){
		memset( heapLine, 0, HEAP_LINE_SIZE);

		fgets( heapLine, HEAP_LINE_SIZE, dataset_file);
		cur_column = tokenize_binary_string( heapLine, 0, &test_nnz); 

		if (max_test_col < cur_column) max_test_col = cur_column; 
		if (minCol > cur_column) minCol = cur_column; 
		
		if (heapLine[0] == 0) break;
		numLines ++;
	}
	max_test_row = numLines; 
	fclose( dataset_file ); 

	fprintf( stderr, "Done with reading %d points from the input files ....(%d, %d ), NNZ: %d, %d \n", 
			numLines, max_test_row, max_test_col, test_nnz, minCol ); 

	if (max_train_col < max_test_col ){
		fprintf (stderr, "Dimensions of Train -- %d, %d \n", max_train_row, max_test_col );
		fprintf (stderr, "Dimensions of Test -- %d, %d \n", max_test_row, max_test_col );
	} else {
		fprintf (stderr, "Dimensions of Train -- %d, %d \n", max_train_row, max_train_col );
		fprintf (stderr, "Dimensions of Test -- %d, %d \n", max_test_row, max_train_col );
	}

	//Read the matrices Here. 
	train_row_id = (int *) malloc ( train_nnz * sizeof (int) ); 
	train_col_id = (int *) malloc ( train_nnz * sizeof (int) ); 
	train_val = (real *) malloc ( train_nnz * sizeof (real) ); 
	train_vec = (real *) malloc ( max_train_row * sizeof (real) ); 

	if ( (dataset_file = fopen(f_train_features, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}

	rowPtr = train_row_id; 
	colPtr = train_col_id; 
	valPtr = train_val; 
	labelPtr = train_vec; 
	numLines = 0; 
	while (!feof( dataset_file) ){
		memset( line, 0, MAX_LINE );
		cols[0] = cols[1] = cols[2] = 0; 

		fgets( line, MAX_LINE, dataset_file);
		if (line[0] == 0) break;

		rowNNZ = tokenize_binary_populate( line, 0, rowPtr, colPtr, valPtr, labelPtr, numLines ); 
		rowPtr += rowNNZ; 
		colPtr += rowNNZ; 
		valPtr += rowNNZ; 

		numLines ++;
	}
	fclose( dataset_file ); 

	for (int i = 0; i < numLines; i ++) 
		if (train_vec[i] == -1) train_vec[i] = 2; 

	fprintf( stderr, "Done populating the training part ... \n"); 

	//Read the test dataset here. 
	test_row_id = (int *) malloc ( test_nnz * sizeof (int) ); 
	test_col_id = (int *) malloc ( test_nnz * sizeof (int) ); 
	test_val = (real *) malloc ( test_nnz * sizeof (real) ); 
	test_vec = (real *) malloc ( max_test_row * sizeof (real) ); 

	if ( (dataset_file = fopen(f_test_features, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}

	rowPtr = test_row_id; 
	colPtr = test_col_id; 
	valPtr = test_val; 	
	labelPtr = test_vec; 

	numLines = 0; 
	while (!feof( dataset_file) ){
		memset( line, 0, MAX_LINE );
		cols[0] = cols[1] = cols[2] = 0; 

		fgets( line, MAX_LINE, dataset_file);
		rowNNZ = tokenize_binary_populate( line, 0, rowPtr, colPtr, valPtr, labelPtr, numLines); 
		rowPtr += rowNNZ; 
		colPtr += rowNNZ; 
		valPtr += rowNNZ; 
	
		if (line[0] == 0) break;

		numLines ++;
	}
	fclose( dataset_file ); 
	for (int i = 0; i < numLines; i ++) 
		if (test_vec[i] == -1) test_vec[i] = 2; 

	fprintf( stderr, "Done populating the testing part ... \n"); 

	//form the cuSparseMatrix Here. 
	data->trainRowPtr = train_row_id; 
	data->trainColPtr = train_col_id; 
	data->trainValPtr = train_val; 
	data->trainLabels = train_vec; 

	data->testRowPtr = test_row_id; 
	data->testColPtr = test_col_id; 
	data->testValPtr = test_val; 
	data->testLabels = test_vec; 

	data->numclasses = 1; 
	data->rows = max_test_row + max_train_row; 
	data->trainSize = max_train_row; 
	data->testSize = max_test_row; 

	data->trainNNZ = train_nnz; 
	data->testNNZ = test_nnz;

	data->trainSet = NULL; 
	data->testSet = NULL; 

	if (max_train_col < max_test_col )
		data->cols = max_test_col; 
	else 
		data->cols = max_train_col; 

	data->trainSet = NULL; 
	data->testSet = NULL; 

	free(heapLine );
}

void readNewsgroupsDataset( char *f_train_features, char *f_train_labels, 
				char *f_test_features, char *f_test_labels, 
				ForestDataset *data, SCRATCH_AREA *s, int offset)
{
	FILE *dataset_file; 
	char line[MAX_LINE]; 
	int numLines = 0;
	int NUM_CLASSES = 20;
	size_t output; 
	int idx = 0; 
	int i;
	real cols[3]; 
	int max_train_col, max_test_col; 
	int max_train_row, max_test_row; 

	char filename[MAX_LINE]; 

	real *scratch = s->hostWorkspace;

	int *train_row_id, *train_col_id; 
	int *test_row_id, *test_col_id; 

	real *train_val, *train_vec; 
	real *test_val, *test_vec; 
	
	int train_nnz, test_nnz; 

	
	if ( (dataset_file = fopen(f_train_features, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}

	max_train_row = max_train_col = 0; 
	while (!feof( dataset_file) ){
		memset( line, 0, MAX_LINE );
		cols[0] = cols[1] = cols[2] = 0; 

		fgets( line, MAX_LINE, dataset_file);
		if (line[0] == 0) break;

		tokenize_string( line, cols, 0 ); 

		if (max_train_row < cols[0]) max_train_row = cols[0]; 
		if (max_train_col < cols[1]) max_train_col = cols[1]; 
		
		numLines ++;
	}
	train_nnz = numLines; 
	fclose( dataset_file ); 
	fprintf( stderr, "Done with reading %d points from the input files ....(%d, %d) \n", 
			numLines, max_train_row, max_train_col ); 


	if ( (dataset_file = fopen(f_test_features, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}

	max_test_row = max_test_col = numLines = 0; 
	while (!feof( dataset_file) ){
		memset( line, 0, MAX_LINE );
		cols[0] = cols[1] = cols[2] = 0; 

		fgets( line, MAX_LINE, dataset_file);
		tokenize_string( line, cols, 0 ); 

		if (max_test_row < cols[0]) max_test_row = cols[0]; 
		if (max_test_col < cols[1]) max_test_col = cols[1]; 
		
		if (line[0] == 0) break;
		numLines ++;
	}
	test_nnz = numLines; 
	fclose( dataset_file ); 

	fprintf( stderr, "Done with reading %d points from the input files ....(%d, %d) \n", 
			numLines, max_test_row, max_test_col ); 

	if (max_train_col < max_test_col ){
		fprintf (stderr, "Dimensions of Train -- %d, %d \n", max_train_row, max_test_col );
		fprintf (stderr, "Dimensions of Test -- %d, %d \n", max_test_row, max_test_col );
	} else {
		fprintf (stderr, "Dimensions of Train -- %d, %d \n", max_train_row, max_train_col );
		fprintf (stderr, "Dimensions of Test -- %d, %d \n", max_test_row, max_train_col );
	}

	//Read the matrices Here. 
	train_row_id = (int *) malloc ( train_nnz * sizeof (int) ); 
	train_col_id = (int *) malloc ( train_nnz * sizeof (int) ); 
	train_val = (real *) malloc ( train_nnz * sizeof (real) ); 
	train_vec = (real *) malloc ( max_train_row * sizeof (real) ); 

	if ( (dataset_file = fopen(f_train_features, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}

	numLines = 0; 
	while (!feof( dataset_file) ){
		memset( line, 0, MAX_LINE );
		cols[0] = cols[1] = cols[2] = 0; 

		fgets( line, MAX_LINE, dataset_file);
		if (line[0] == 0) break;

		tokenize_string( line, cols, 0 ); 

		train_row_id[ numLines ] = (int)(cols[0] - 1); 
		train_col_id[ numLines ] = (int)(cols[1] - 1); 
		train_val[ numLines ] = (real)cols[2]; 

		//fprintf( stderr, " %d, %d, %f \n", train_row_id[ numLines ], train_col_id[ numLines ], train_val [numLines ] ); 

		numLines ++;
	}
	fclose( dataset_file ); 

	//vector here. 
	i = readVector( train_vec, max_train_row, f_train_labels, offset ); 
	fprintf( stderr, "Labels read from file: %d, expected : %d \n", i, max_train_row ); 

	//compute the NUM_CLASSES Here. 
	NUM_CLASSES = computeMaxValue( train_vec, max_train_row); 

	//Read the test dataset here. 
	test_row_id = (int *) malloc ( test_nnz * sizeof (int) ); 
	test_col_id = (int *) malloc ( test_nnz * sizeof (int) ); 
	test_val = (real *) malloc ( test_nnz * sizeof (real) ); 
	test_vec = (real *) malloc ( max_test_row * sizeof (real) ); 

	if ( (dataset_file = fopen(f_test_features, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}

	numLines = 0; 
	while (!feof( dataset_file) ){
		memset( line, 0, MAX_LINE );
		cols[0] = cols[1] = cols[2] = 0; 

		fgets( line, MAX_LINE, dataset_file);
		tokenize_string( line, cols, 0 ); 

		if (line[0] == 0) break;
		test_row_id[ numLines ] = (int)(cols[0] - 1); 
		test_col_id[ numLines ] = (int)(cols[1] - 1); 
		test_val[ numLines ] = (real)cols[2]; 

		numLines ++;
	}
	fclose( dataset_file ); 

	//vector here. 
	i = readVector( test_vec, max_test_row, f_test_labels, offset ); 

	//form the cuSparseMatrix Here. 
	data->trainRowPtr = train_row_id; 
	data->trainColPtr = train_col_id; 
	data->trainValPtr = train_val; 
	data->trainLabels = train_vec; 

	data->testRowPtr = test_row_id; 
	data->testColPtr = test_col_id; 
	data->testValPtr = test_val; 
	data->testLabels = test_vec; 

	data->numclasses = NUM_CLASSES - 1; 
	data->rows = max_test_row + max_train_row; 
	data->trainSize = max_train_row; 
	data->testSize = max_test_row; 

	data->trainNNZ = train_nnz; 
	data->testNNZ = test_nnz;

	data->trainSet = NULL; 
	data->testSet = NULL; 

	if (max_train_col < max_test_col )
		data->cols = max_test_col; 
	else 
		data->cols = max_train_col; 

	data->trainSet = NULL; 
	data->testSet = NULL; 

	// preprocess the dataset here. 
	/*
	real train_max = findMaxInDataset( data->trainValPtr, data->trainNNZ, 1 ); 
	real test_max = findMaxInDataset( data->testValPtr, data->testNNZ, 1 ); 
	fprintf( stderr, "Train max: %f, Test max: %f \n", train_max, test_max ); 

	if (train_max < test_max){
		preprocessDataset ( data->trainValPtr, data->trainNNZ, 1, test_max ); 
		preprocessDataset ( data->testValPtr, data->testNNZ, 1, test_max ); 
	} else {
		preprocessDataset ( data->trainValPtr, data->trainNNZ, 1, train_max ); 
		preprocessDataset ( data->testValPtr, data->testNNZ, 1, train_max ); 
	}
	*/
}

void readCIFARDataset( char *dir, char *train, char *test, ForestDataset *data, SCRATCH_AREA *s, int raw) {

	FILE *dataset_file; 
	char line[MAX_LINE]; 
	int numLines = 0;
	int NUM_CLASSES = 10;
	size_t output; 
	int idx = 0; 
	int i;
	int TRAIN_IMAGES = 50000; 
	int TRAIN_FILES = 5; 

	char filename[MAX_LINE]; 
	real *train_set, *train_labels, *test_set, *test_labels;
	real *scratch = s->hostWorkspace;

	train_set = (real *) malloc( (size_t)TRAIN_IMAGES * (CIFAR_LINE_SIZE-1) * sizeof(real) );
	train_labels = (real *) malloc ( (size_t)TRAIN_IMAGES * sizeof(real) ); 
	test_set = (real *) malloc( (size_t)10000 * (CIFAR_LINE_SIZE-1) * sizeof(real) );
	test_labels = (real *) malloc ( (size_t)10000 * sizeof(real) ); 

	fprintf( stderr, " Allocated memory for the dataset : %lu \n", TRAIN_IMAGES * (CIFAR_LINE_SIZE-1) * sizeof(real)); 
	fprintf( stderr, " Allocated memory for the dataset (GB): %d \n", (TRAIN_IMAGES * (CIFAR_LINE_SIZE-1) * sizeof(real)) / (1024 * 1024 * 1024)); 

	numLines = 0; 
	for (idx = 1; idx <= TRAIN_FILES; idx ++) {	
		sprintf( filename, "%s%s%d.bin", dir, train, idx); 
		fprintf( stderr, "Reading file : %s \n", filename ); 

		if ( (dataset_file = fopen(filename, "r")) == NULL ) { 
			fprintf( stderr, "Error opening the dataset.... !\n" );
			exit( -1 );
		}

		while (!feof( dataset_file) ){
			memset( line, 0, MAX_LINE );
			output = fread( line, (size_t)1, (size_t)CIFAR_LINE_SIZE, dataset_file);

			if (output <= 0) break;

			train_labels[ numLines ] = line[0] + 1;  
			for (i = 0; i < CIFAR_LINE_SIZE-1; i ++) 
				train_set[ numLines * (CIFAR_LINE_SIZE - 1) + i ] = (unsigned char) line[i + 1]; 

			numLines ++;
		}
	}
	fprintf( stderr, "Done with reading %d points from the input files .... \n", numLines ); 

	//test data here. 
	numLines = 0; 
	memset( filename, 0, MAX_LINE ); 
	sprintf( filename, "%s%s", dir, test); 

	if ( (dataset_file = fopen(filename, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}

	while (!feof( dataset_file) ){
		memset( line, 0, MAX_LINE );
		output = fread( line, (size_t)1, (size_t)CIFAR_LINE_SIZE, dataset_file);
		if (output <= 0) break;

		test_labels[ numLines ] = line[0] + 1; 
		for (i = 0; i < CIFAR_LINE_SIZE - 1; i ++) 
			test_set[ numLines * (CIFAR_LINE_SIZE - 1) + i ] = (unsigned char) line[i + 1]; 

		numLines ++;
	}
	fprintf( stderr, "Done with reading %d points from the input files .... \n", numLines ); 

	//inititalize the device data here. 
	data->trainSize = TRAIN_IMAGES; 
	data->testSize = 10000; 
	data->trainSet = train_set;
	data->trainLabels = train_labels;
	data->testSet = test_set; 
	data->testLabels = test_labels;
	data->rows = data->trainSize + data->testSize;
	data->cols = CIFAR_LINE_SIZE - 1;
	data->numclasses = NUM_CLASSES - 1; 

	data->trainRowPtr = NULL; 
	data->trainColPtr = NULL; 
	data->trainValPtr = NULL; 

	data->testRowPtr = NULL; 
	data->testColPtr = NULL; 
	data->testValPtr = NULL; 

/*
	fprintf(stderr, "Preprocessing .... \n"); 
	real train_max = findMaxInDataset( train_set, data->trainSize, data->cols ); 
	real test_max = findMaxInDataset( test_set, data->testSize, data->cols ); 
	fprintf( stderr, "TrainMax %e and TestMax: %e \n", train_max, test_max ); 

	if (train_max >= test_max) {
		preprocessDataset( train_set, data->trainSize, data->cols, train_max );
		preprocessDataset( test_set, data->testSize, data->cols, train_max );
	} else {
		preprocessDataset( train_set, data->trainSize, data->cols, test_max );
		preprocessDataset( test_set, data->testSize, data->cols, test_max );
	}
*/

	fprintf( stderr, "Converting to column major format here.... \n"); 
	//train_features
	convertToColumnMajor( train_set, data->trainSize, data->cols, scratch);
	fprintf( stderr, "Done with conversion... \n"); 
	memcpy( train_set, scratch, (size_t)(sizeof(real) * data->trainSize * data->cols) );

	//test_features
	convertToColumnMajor( test_set, data->testSize, data->cols, scratch);
	fprintf( stderr, "Done with conversion... \n"); 
	memcpy( test_set, scratch, (size_t)(sizeof(real) * data->testSize * data->cols) );

        if (raw == 0){
                fprintf( stderr, "Normalizing the data ... ");
                columnNormalize( train_set, data->trainSize, data->cols, test_set, data->testSize );
                fprintf( stderr, "Done... \n");
        }

}

void readMultiDataset( char *f_train_features, char *f_train_labels, 
		char *f_test_features, char *f_test_labels, ForestDataset *data, SCRATCH_AREA *s, int offset, int bias)
{
	FILE *dataset_file;
	char line[MAX_LINE];
	int numLines = 0;
	real temp[MAX_LINE]; 
	int NUM_CLASSES = -1;

	real *train_set, *train_labels, *test_set, *test_labels;
	real *scratch = s->hostWorkspace;

	if ( (dataset_file = fopen(f_train_features, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}

	while (!feof( dataset_file) ){
		memset( line, 0, MAX_LINE );
		fgets( line, MAX_LINE, dataset_file);

		if (line[0] == 0) break;
		//data->cols = tokenize_learn_multiclass( line, temp, numLines, NULL, NULL);
		data->cols = tokenize_string( line, temp, bias);
		numLines ++;
	}

	fprintf(stderr, "Number of columns is : %d \n", data->cols ); 
	fprintf( stderr, "Train Size: %d \n", numLines ); 

	//exit (-1); 

	data->trainSize = numLines;
	/*
	train_set = (real *)malloc( (FEATURE_SIZE_MULTI) * data->trainSize);
	train_labels = (real *)malloc(sizeof(real) * data->trainSize);
	*/
	train_set = (real *)malloc(  data->cols * data->trainSize * sizeof(real));
	train_labels = (real *)malloc( data->trainSize * sizeof(real));

	//read the file here and fill the matrix. 
	rewind( dataset_file );	
	numLines = 0;

	while (!feof( dataset_file )){
		memset( line, 0, MAX_LINE );
		fgets( line, MAX_LINE, dataset_file);
		if (line[0] == 0) break;
		tokenize_populate( line, train_set, &numLines, data->cols, bias );
		numLines ++;
	}
	fclose( dataset_file );

	//read the train labels here. 
	fprintf( stderr, " Reading the vector: %s \n", f_train_labels ); 
	readVector( train_labels, data->trainSize, f_train_labels, offset ); 

	//compute the NUM_CLASSES Here. 
	NUM_CLASSES = computeMaxValue( train_labels, data->trainSize ); 

	//read the test dataset here. 
	fprintf( stderr, " Reading the test Matrix: %s \n", f_test_features ); 
	if ( (dataset_file = fopen(f_test_features, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the dataset.... !\n" );
		exit( -1 );
	}

	numLines = 0; 
	while (!feof( dataset_file) ){
		memset( line, 0, MAX_LINE );
		fgets( line, MAX_LINE, dataset_file);

		if (line[0] == 0) break;
		//data->cols = tokenize_learn_multiclass( line, temp, numLines, NULL, NULL);
		data->cols = tokenize_string( line, temp, bias );
		numLines ++;
	}

	fprintf(stderr, "Test size: %d \n", numLines ); 
	fprintf( stderr, "Number of features for test set: %d \n", data->cols ); 

	data->testSize = numLines;
	/*
	test_set = (real *)malloc( (FEATURE_SIZE_MULTI) * data->testSize);
	test_labels = (real *)malloc(sizeof(real) * data->testSize);
	*/
	test_set = (real *)malloc( data->cols * data->testSize *  sizeof(real));
	test_labels = (real *)malloc(data->testSize *  sizeof(real));

	//read the test set
	rewind( dataset_file );	
	numLines = 0;

	while (!feof( dataset_file )){
		memset( line, 0, MAX_LINE );
		fgets( line, MAX_LINE, dataset_file);
		if (line[0] == 0) break;
		tokenize_populate( line, test_set, &numLines, data->cols, bias );
		numLines ++;
	}
	fclose( dataset_file );

	//read the test labels here. 
	readVector( test_labels, numLines, f_test_labels, offset ); 
	real testMax = computeMaxValue( test_labels, numLines );
	if (testMax > NUM_CLASSES) 
		NUM_CLASSES = (int) testMax; 
		

	//initialization here. 
	data->trainSet = train_set;
	data->trainLabels = train_labels;
	data->testSet = test_set; 
	data->testLabels = test_labels;
	data->rows = data->trainSize + data->testSize;
	data->numclasses = NUM_CLASSES - 1; 

	data->trainRowPtr = NULL; 
	data->trainColPtr = NULL; 
	data->trainValPtr = NULL; 

	data->testRowPtr = NULL; 
	data->testColPtr = NULL; 
	data->testValPtr = NULL; 

	//preprocessing step here. 
	/*
	real train_max = findMaxInDataset( train_set, data->trainSize, data->cols ); 
	real test_max = findMaxInDataset( test_set, data->testSize, data->cols ); 

	if (train_max >= test_max) {
		preprocessDataset( train_set, data->trainSize, data->cols, train_max );
		preprocessDataset( test_set, data->testSize, data->cols, train_max );
	} else {
		preprocessDataset( train_set, data->trainSize, data->cols, test_max );
		preprocessDataset( test_set, data->testSize, data->cols, test_max );
	}
	*/	

	//train_features
	convertToColumnMajor( train_set, data->trainSize, data->cols, scratch);
	memcpy( train_set, scratch, sizeof(real) * data->trainSize * data->cols );

	//test_features
	convertToColumnMajor( test_set, data->testSize, data->cols, scratch);
	memcpy( test_set, scratch, sizeof(real) * data->testSize * data->cols );

	//DEBUG HERE. 
	/*
	fprintf (stderr, "Train Set Here \n"); 
	for (int i = 0; i < data->trainSize; i ++) 
		fprintf( stderr, " %2.2f ", train_set[ i ] ); 
	fprintf( stderr, "\n"); 

	fprintf( stderr, "Labels here \n"); 
	for (int i = 0; i < data->trainSize; i ++)
		fprintf( stderr, " %2.2f ", train_labels[ i ] ); 
	fprintf (stderr, "\n"); 
	*/	
}

int tokenize_binary_populate( char *line, int bias, int *row, int *col, real *val, real *label, int rowNum )
{
	const char *sep = ", \n";
	char *word, *ptr;
	char temp[MAX_LINE];
	int index = 0; 
	int len = 0; 

	char col_str[32]; 

	if (bias >= 1){ 
		*row = rowNum; row ++; 
		*col = 0; col ++;
		*val = 1; val ++; 

		index = 1;
	}

	strncpy( temp, line, MAX_LINE );
	for( word = strtok(temp, sep); word; word = strtok(NULL, sep) )
	{ 
		memset( col_str, 0, sizeof(char) * 32);
		memcpy( col_str, word, 31); 
		len = 0; 

		ptr = col_str; 
		while (*ptr != 0 && *ptr != ':'){ 
			ptr ++; 
			len ++; 
		}

		if (*ptr == ':') {
			*ptr = 0; 

			*row = rowNum; row ++; 
			*col = atoi( col_str) - 1;  col ++; 
			*val = atof( col_str + len + 1);  val ++; 
			index ++; 

		} else {
			label[rowNum] = atof( word ); 
		}
	}

	return index;
}

int tokenize_binary_string( char *line, int bias, int *nnz)
{
	const char *sep = ", \n";
	char *word, *ptr;
	//char temp[MAX_LINE];
	int index = 0; 
	int col = 0; 
	real val = 0; 
	int len = 0; 

	char col_str[32];

	if (bias >= 1) index = 1;

	for( word = strtok(line, sep); word; word = strtok(NULL, sep) )
	{ 
		col = val = -99; 
		memset( col_str, 0, 32);

		strncpy( col_str, word, 31 ); 
		ptr = col_str; 

		len = 0; 
		while (*ptr != 0 && *ptr != ':'){ 
			ptr ++; 
			len ++; 
		}

		if (*ptr == ':') {
			*ptr = 0; 
			col = atoi( col_str ); // to account for zero here.
			val = atof( col_str + len + 1 ); 

			(*nnz) ++; 
		}
	}

	return col;
}


int tokenize_string( char *line, real *out, int bias )
{
	const char *sep = ", \n";
	char *word;
	char temp[MAX_LINE];
	int index = 0; 

	if (bias >= 1) index = 1;

	strncpy( temp, line, MAX_LINE );
	for( word = strtok(temp, sep); word; word = strtok(NULL, sep) ) out[ index ++] = atof( word );

	return index;
}

void tokenize_populate(char *line, real *train_set, int *count, int size, int bias){

	const char *sep = ", \n";
	char *word;
	char temp[MAX_LINE];
	int index = 0; 
	real cur_row[MAX_LINE]; 

	if (bias >= 1) cur_row[ index ++ ] = 1;

	strncpy( temp, line, MAX_LINE );
	for( word = strtok(temp, sep); word; word = strtok(NULL, sep) ) cur_row[ index ++] = atof( word );
	memcpy( &train_set[ (*count) * (size)], cur_row, sizeof(real) * size);
}


void printDataset( ForestDataset *t)
{
	fprintf( stderr, "--------------------");
	fprintf( stderr, "Train Row 1: ");
	for (int i = 0; i < 52; i ++)
		fprintf( stderr, " %f ", t->trainSet[i] );
	fprintf( stderr, "\n");
	fprintf( stderr, "Test Row 1: ");
	for (int i = 0; i < 52; i ++)
		fprintf( stderr, " %f ", t->testSet[i] );
	fprintf( stderr, "\n");

	fprintf( stderr, "Train Labels: \n");
	for (int i = 0; i < t->rows; i ++)
		fprintf (stderr, " %f ", t->trainLabels[i] );
	fprintf( stderr, "\n");

	fprintf( stderr, "Test Labels: \n");
	for (int i = 0; i < 200; i ++)
		fprintf (stderr, " %f ", t->testLabels[i] );
	fprintf( stderr, "\n");
	fprintf( stderr, "--------------------\n");
}

//
//
// Device Functions here. 
//
//
void initialize_device_data( ForestDataset *s, DeviceDataset *t)
{
	t->rows = s->trainSize;
	t->cols = s->cols;
	t->testSize = s->testSize;
	t->numclasses = s->numclasses; 

	cuda_malloc( (void **)&t->trainSet, t->rows * t->cols * sizeof(real), 0, ERROR_MEM_ALLOC );
	copy_host_device( s->trainSet, t->trainSet, t->rows * t->cols * sizeof(real), cudaMemcpyHostToDevice, ERROR_MEMCPY_TRAINSET );

	cuda_malloc( (void **)&t->trainLabels, t->rows  * sizeof(real), 0, ERROR_MEM_ALLOC );
	copy_host_device( s->trainLabels, t->trainLabels, t->rows * sizeof(real), cudaMemcpyHostToDevice, ERROR_MEMCPY_TRAINLABELS );
	
	cuda_malloc( (void **)&t->testSet, t->testSize * t->cols * sizeof(real), 0, ERROR_MEM_ALLOC );
	copy_host_device( s->testSet, t->testSet, t->testSize * t->cols * sizeof(real), cudaMemcpyHostToDevice, ERROR_MEMCPY_TESTSET );

	cuda_malloc( (void **)&t->testLabels, t->testSize  * sizeof(real), 0, ERROR_MEM_ALLOC );
	copy_host_device( s->testLabels, t->testLabels, t->testSize * sizeof(real), cudaMemcpyHostToDevice, ERROR_MEMCPY_TESTLABELS );

	if (t->numclasses > 1)
		cuda_malloc( (void **)&t->weights, t->numclasses * t->cols * sizeof(real), 1, ERROR_MEM_ALLOC );
	else
		cuda_malloc( (void **)&t->weights, t->cols * sizeof(real), 1, ERROR_MEM_ALLOC );

#ifdef __debug__
	fprintf (stderr, " -------------- \n");
	fprintf( stderr, "Train Set size: %d %d, %d \n", t->rows, t->cols, t->testSize );
	fprintf (stderr, " -------------- \n");
#endif

	t->spTrain.rowPtr = NULL; 
	t->spTrain.colPtr = NULL; 
	t->spTrain.valPtr = NULL; 
	t->spTrain.rowCsrPtr = NULL; 

	t->spTest.rowPtr = NULL; 
	t->spTest.colPtr = NULL; 
	t->spTest.valPtr = NULL; 
	t->spTest.rowCsrPtr = NULL; 

	//printVector (t->testSet, t->testSize, NULL);
	//printVector( t->trainLabels, t->rows, s->trainLabels );
	

	//sub sampling here. 
	//Hesian part here
	t->spSampledHessianTrain.nnz = 0; 
	t->spSampledHessianTrain.P = NULL; 
	t->spSampledHessianTrain.sortedVals= NULL; 
	t->spSampledHessianTrain.rowPtr= NULL;
	t->spSampledHessianTrain.colPtr= NULL; 
	t->spSampledHessianTrain.valPtr= NULL; 
	t->spSampledHessianTrain.rowCsrPtr= NULL; 

	t->hessianSampleSize = (SAMPLING_BUFFER_EXTENSION * HESSIAN_SAMPLING_SIZE * t->rows) / 100;
	t->spHessianSample.nnz = t->hessianSampleSize; 
	cuda_malloc( (void **)&t->sampledHessianTrainSet, t->hessianSampleSize* t->cols * sizeof(real), 0, ERROR_MEM_ALLOC );
	//fprintf( stderr, "SubSampled Size for this dataset (Hessian): %d \n", t->hessianSampleSize); 

	//spSampledHessian
	cuda_malloc( (void **) &t->spHessianSample.P, 		t->hessianSampleSize * sizeof(int), 1, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spHessianSample.sortedVals,  t->hessianSampleSize * sizeof(real), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spHessianSample.rowPtr, 	t->hessianSampleSize * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spHessianSample.colPtr, 	t->hessianSampleSize * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spHessianSample.valPtr, 	t->hessianSampleSize * sizeof(real), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spHessianSample.rowCsrPtr,   (t->hessianSampleSize + 1) * sizeof(int), 1, ERROR_MEM_ALLOC ); 

	//Gradient Sample Here. 
	t->spSampledGradientTrain.nnz = 0; 
	t->spSampledGradientTrain.P = NULL; 
	t->spSampledGradientTrain.sortedVals = NULL; 
	t->spSampledGradientTrain.rowPtr = NULL; 
	t->spSampledGradientTrain.colPtr = NULL; 
	t->spSampledGradientTrain.valPtr = NULL; 
	t->spSampledGradientTrain.rowCsrPtr = NULL; 

	t->gradientSampleSize = (GRADIENT_SAMPLING_SIZE * t->rows ) / 100; 
	t->spGradientSample.nnz = t->gradientSampleSize; 
	cuda_malloc( (void **)&t->sampledGradientTrainSet, t->gradientSampleSize* t->cols * sizeof(real), 0, ERROR_MEM_ALLOC );
	cuda_malloc( (void **)&t->sampledGradientTrainLabels, t->gradientSampleSize * sizeof(real), 0, ERROR_MEM_ALLOC );
	fprintf( stderr, "SubSampled Size for this dataset (Gradient): %d \n", t->gradientSampleSize); 

	//spSampledHessian
	cuda_malloc( (void **) &t->spGradientSample.P, 		t->gradientSampleSize * sizeof(int), 1, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spGradientSample.sortedVals,  t->gradientSampleSize * sizeof(real), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spGradientSample.rowPtr, 	t->gradientSampleSize * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spGradientSample.colPtr, 	t->gradientSampleSize * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spGradientSample.valPtr, 	t->gradientSampleSize * sizeof(real), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spGradientSample.rowCsrPtr,   (t->gradientSampleSize + 1) * sizeof(int), 1, ERROR_MEM_ALLOC ); 

		
}

void initialize_device_data_sparse( ForestDataset *s, DeviceDataset *t )
{

	t->trainSet = NULL; 
	t->testSet = NULL; 
	
	t->rows = s->trainSize;
	t->cols = s->cols;
	t->testSize = s->testSize;
	t->numclasses = s->numclasses; 

	//t->trainNNZ = s->trainNNZ; 
	//t->testNNZ = s->testNNZ; 
	t->spTrain.nnz = s->trainNNZ; 
	t->spTest.nnz = s->testNNZ; 

	fprintf( stderr, "NNZ: %d, %d \n", s->trainNNZ, s->testNNZ ); 

	//Train Set Here. 
	cuda_malloc( (void **) &t->spTrain.rowPtr, s->trainNNZ * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	copy_host_device( s->trainRowPtr, t->spTrain.rowPtr, s->trainNNZ * sizeof(int), cudaMemcpyHostToDevice, ERROR_MEMCPY_TRAINSET ); 

	cuda_malloc( (void **) &t->spTrain.colPtr, s->trainNNZ * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	copy_host_device( s->trainColPtr, t->spTrain.colPtr, s->trainNNZ * sizeof(int), cudaMemcpyHostToDevice, ERROR_MEMCPY_TRAINSET ); 

	cuda_malloc( (void **) &t->spTrain.valPtr, s->trainNNZ * sizeof(real), 0, ERROR_MEM_ALLOC ); 
	copy_host_device( s->trainValPtr, t->spTrain.valPtr, s->trainNNZ * sizeof(real), cudaMemcpyHostToDevice, ERROR_MEMCPY_TRAINSET ); 

	cuda_malloc( (void **) &t->trainLabels, t->rows * sizeof(real), 0, ERROR_MEM_ALLOC ); 
	copy_host_device( s->trainLabels, t->trainLabels, t->rows* sizeof(real), cudaMemcpyHostToDevice, ERROR_MEMCPY_TRAINSET ); 

	cuda_malloc( (void **) &t->spTrain.rowCsrPtr, (t->rows + 1) * sizeof(int), 1, ERROR_MEM_ALLOC ); 

	//allocate the csc format matrix space here. 
	//cuda_malloc( (void **) &t->spTrain.cscRowPtr, s->trainNNZ * sizeof(int), 1, ERROR_MEM_ALLOC ); 
	//cuda_malloc( (void **) &t->spTrain.cscColPtr, (t->cols + 1) * sizeof(int), 1, ERROR_MEM_ALLOC ); 
	//cuda_malloc( (void **) &t->spTrain.cscValPtr, s->trainNNZ * sizeof(double), 1, ERROR_MEM_ALLOC ); 

	//allocate the data for sorted coo format here. 
	cuda_malloc( (void **) &t->spTrain.P, s->trainNNZ * sizeof(int), 1, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spTrain.sortedVals, s->trainNNZ * sizeof(real), 0, ERROR_MEM_ALLOC ); 

	

	fprintf( stderr, "Done with training .... \n"); 

	//TestSet Here. 
	cuda_malloc( (void **) &t->spTest.rowPtr, s->testNNZ * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	copy_host_device( s->testRowPtr, t->spTest.rowPtr, s->testNNZ * sizeof(int), cudaMemcpyHostToDevice, ERROR_MEMCPY_TRAINSET ); 

	cuda_malloc( (void **) &t->spTest.colPtr, s->testNNZ * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	copy_host_device( s->testColPtr, t->spTest.colPtr, s->testNNZ * sizeof(int), cudaMemcpyHostToDevice, ERROR_MEMCPY_TRAINSET ); 

	cuda_malloc( (void **) &t->spTest.valPtr, s->testNNZ * sizeof(real), 0, ERROR_MEM_ALLOC ); 
	copy_host_device( s->testValPtr, t->spTest.valPtr, s->testNNZ * sizeof(real), cudaMemcpyHostToDevice, ERROR_MEMCPY_TRAINSET ); 

	cuda_malloc( (void **) &t->testLabels, t->testSize * sizeof(real), 0, ERROR_MEM_ALLOC ); 
	copy_host_device( s->testLabels, t->testLabels, t->testSize * sizeof(real), cudaMemcpyHostToDevice, ERROR_MEMCPY_TRAINSET ); 

	cuda_malloc( (void **) &t->spTest.rowCsrPtr, (t->testSize + 1) * sizeof(int), 1, ERROR_MEM_ALLOC ); 

	//allocate the data for sorted coo format here. 
	cuda_malloc( (void **) &t->spTest.P, s->testNNZ * sizeof(int), 1, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spTest.sortedVals, s->testNNZ * sizeof(real), 0, ERROR_MEM_ALLOC ); 

	fprintf( stderr, "Done with testing .... \n"); 

	//Weights Here. 
	if (t->numclasses > 1)
		cuda_malloc( (void **)&t->weights, t->numclasses * t->cols * sizeof(real), 1, ERROR_MEM_ALLOC );
	else
		cuda_malloc( (void **)&t->weights, t->cols * sizeof(real), 1, ERROR_MEM_ALLOC );

	//sparse sample matrices here. 
	//sub sampling here. 
	//Hesian part here
	t->sampledGradientTrainSet = NULL; 
	t->sampledHessianTrainSet = NULL; 
	t->hessianSampleSize = (SAMPLING_BUFFER_EXTENSION * HESSIAN_SAMPLING_SIZE * t->rows) / 100;
	t->spHessianSample.nnz = t->hessianSampleSize; 

	//spSampledHessian
	cuda_malloc( (void **) &t->spHessianSample.P, 		t->hessianSampleSize * sizeof(int), 1, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spHessianSample.sortedVals,  t->hessianSampleSize * sizeof(real), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spHessianSample.rowPtr, 	t->hessianSampleSize * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spHessianSample.colPtr, 	t->hessianSampleSize * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spHessianSample.valPtr, 	t->hessianSampleSize * sizeof(real), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spHessianSample.rowCsrPtr,   (t->hessianSampleSize + 1) * sizeof(int), 1, ERROR_MEM_ALLOC ); 

	//Gradient Sample Here. 
	t->gradientSampleSize = (GRADIENT_SAMPLING_SIZE * t->rows ) / 100; 
	t->spGradientSample.nnz = t->gradientSampleSize; 
	cuda_malloc( (void **)&t->sampledGradientTrainLabels, t->gradientSampleSize * sizeof(real), 0, ERROR_MEM_ALLOC );

	//spSampledHessian
	cuda_malloc( (void **) &t->spGradientSample.P, 		t->gradientSampleSize * sizeof(int), 1, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spGradientSample.sortedVals,  t->gradientSampleSize * sizeof(real), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spGradientSample.rowPtr, 	t->gradientSampleSize * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spGradientSample.colPtr, 	t->gradientSampleSize * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spGradientSample.valPtr, 	t->gradientSampleSize * sizeof(real), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spGradientSample.rowCsrPtr,   (t->gradientSampleSize + 1) * sizeof(int), 1, ERROR_MEM_ALLOC ); 

	
	cuda_malloc( (void **) &t->spSampledGradientTrain.rowPtr, 	s->trainNNZ * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spSampledGradientTrain.colPtr, 	s->trainNNZ * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spSampledGradientTrain.valPtr, 	s->trainNNZ * sizeof(real), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spSampledGradientTrain.rowCsrPtr, 	(t->rows+ 1) * sizeof(int), 1, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spSampledGradientTrain.P, 		s->trainNNZ * sizeof(int), 1, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spSampledGradientTrain.sortedVals, 	s->trainNNZ * sizeof(real), 0, ERROR_MEM_ALLOC ); 

	cuda_malloc( (void **) &t->spSampledHessianTrain.rowPtr, 	s->trainNNZ * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spSampledHessianTrain.colPtr, 	s->trainNNZ * sizeof(int), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spSampledHessianTrain.valPtr, 	s->trainNNZ * sizeof(real), 0, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spSampledHessianTrain.rowCsrPtr, 	(t->rows+ 1) * sizeof(int), 1, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spSampledHessianTrain.P, 		s->trainNNZ * sizeof(int), 1, ERROR_MEM_ALLOC ); 
	cuda_malloc( (void **) &t->spSampledHessianTrain.sortedVals, 	s->trainNNZ * sizeof(real), 0, ERROR_MEM_ALLOC ); 


	//Debug print statements here. 
	fprintf (stderr, " -------------- \n");
	fprintf( stderr, "Train Set size: %d %d, %d \n", t->rows, t->cols, t->testSize );
	fprintf (stderr, " -------------- \n");
}

void cleanup_dataset( ForestDataset *s, DeviceDataset *t){
	if (s->trainSet) release_memory( (void **)&s->trainSet );
	if (s->trainLabels ) release_memory( (void **)&s->trainLabels );
	if (s->testSet ) release_memory( (void **)&s->testSet );
	if (s->testLabels ) release_memory( (void **)&s->testLabels );

	if (t->trainSet) cuda_free ( t->trainSet, ERROR_MEM_CLEANUP );
	if (t->trainLabels ) cuda_free ( t->trainLabels, ERROR_MEM_CLEANUP );
	if (t->testSet) cuda_free ( t->testSet, ERROR_MEM_CLEANUP );
	if (t->testLabels) cuda_free ( t->testLabels, ERROR_MEM_CLEANUP );

	//sparse functions here. 
	if (t->spTrain.rowPtr || t->spTrain.colPtr || t->spTrain.valPtr)
		cusparseDestroyMatDescr( t->spTrain.descr ); 
	if (t->spTrain.rowPtr) cuda_free( t->spTrain.rowPtr, ERROR_MEM_CLEANUP ); 
	if (t->spTrain.colPtr) cuda_free( t->spTrain.colPtr, ERROR_MEM_CLEANUP ); 
	if (t->spTrain.valPtr) cuda_free( t->spTrain.valPtr, ERROR_MEM_CLEANUP ); 
	if (t->spTrain.rowCsrPtr) cuda_free( t->spTrain.rowCsrPtr, ERROR_MEM_CLEANUP ); 

	if (t->spTest.rowPtr || t->spTest.colPtr || t->spTest.valPtr)
		cusparseDestroyMatDescr( t->spTest.descr ); 
	if (t->spTest.rowPtr) cuda_free( t->spTest.rowPtr, ERROR_MEM_CLEANUP ); 
	if (t->spTest.colPtr) cuda_free( t->spTest.colPtr, ERROR_MEM_CLEANUP ); 
	if (t->spTest.valPtr) cuda_free( t->spTest.valPtr, ERROR_MEM_CLEANUP ); 
	if (t->spTest.rowCsrPtr) cuda_free( t->spTest.rowCsrPtr, ERROR_MEM_CLEANUP ); 
}
