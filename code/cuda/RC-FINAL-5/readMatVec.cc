#ifndef __H_MATVEC__
#define __H_MATVEC__

#include "readMatVec.h"
#include <string.h>

#define MAX_LINE 1024

void readMatVec( char *matrixPath, char *vectorPath, double **matrix, double **vector, int *N){

	//read the CSV file here and create
	//matrix and vector files and pass
	//them back to the main file
	FILE *matFile;
	char line[MAX_LINE];
	int numLines = 0;
	int index = 0;
	double *fileMatrix;
	double *fileVector;

   if ( (matFile = fopen(matrixPath, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the pdb file!\n" );
		exit( -1 );
	}

	while (!feof( matFile ) ){
		memset( line, 0, MAX_LINE );
		fgets( line, MAX_LINE, matFile );
		if (line[0] == 0) break;
		numLines ++;
	}
	fprintf( stderr, " Number of lines read: %d \n", numLines );

	*N = numLines;
	fileMatrix = (double *) malloc( sizeof(double) * (numLines) * (numLines) );

	//read the file here and fill the matrix. 
	rewind( matFile );	
	while (!feof( matFile )){
		memset( line, 0, MAX_LINE );
		fgets( line, MAX_LINE, matFile);
		if (line[0] == 0) break;
		tokenize( line, fileMatrix, &index );
	}

	fclose( matFile );
	fprintf( stderr, "Number of elements: %d\n", index );

	//read teh vector here. 
	fileVector = (double *) malloc (sizeof(double) * (numLines) );
   if ( (matFile = fopen(vectorPath, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the pdb file!\n" );
		exit( -1 );
	}

	index = 0;
	while (!feof( matFile )){
		memset( line, 0, MAX_LINE );
		fgets( line, MAX_LINE, matFile);
		if (line[0] == 0) break;
		fileVector[index ++] = atof( line );
		//fprintf (stderr, "%s --> %f\n", line, atof(line) );
	}
	fprintf( stderr, "------------------\n");
	fclose( matFile );
	fprintf( stderr, "Number of elements: %d\n", index );

	*matrix = fileMatrix;
	*vector = fileVector;
}

void tokenize( char *line, double *matrix, int* index){
	char *sep = ", \n";
	char *word;
	char temp[MAX_LINE];

	strncpy( temp, line, MAX_LINE );
	for( word = strtok(temp, sep); word; word = strtok(NULL, sep) )
		matrix[ (*index) ++ ] = atof( word );
}

void tokenize_count( char *line, int* index){
	char *sep = ", \n";
	char *word;
	char temp[MAX_LINE];

	strncpy( temp, line, MAX_LINE );
	for( word = strtok(temp, sep); word; word = strtok(NULL, sep) )
		(*index) ++;
}

void readVec( char *vectorPath, double **vector, int *N){

	FILE *matFile;
	char line[MAX_LINE];
	int numLines = 0;
	int index = 0;
	double *fileVector;

   if ( (matFile = fopen(vectorPath, "r")) == NULL ) { 
		fprintf( stderr, "Error opening the pdb file!\n" );
		exit( -1 );
	}

	while (!feof( matFile ) ){
		memset( line, 0, MAX_LINE );
		fgets( line, MAX_LINE, matFile );
		if (line[0] == 0) break;
		tokenize_count( line, &numLines );
		break;
	}
	fprintf( stderr, " Number of lines read: %d \n", numLines );

	*N = numLines;
	fileMatrix = (double *) malloc( sizeof(double) * (numLines) );

	//read the file here and fill the matrix. 
	rewind( matFile );	
	while (!feof( matFile )){
		memset( line, 0, MAX_LINE );
		fgets( line, MAX_LINE, matFile);
		if (line[0] == 0) break;
		tokenize( line, fileMatrix, &index );
		break;
	}

	fclose( matFile );
	fprintf( stderr, "Number of elements: %d\n", index );
}
#endif
