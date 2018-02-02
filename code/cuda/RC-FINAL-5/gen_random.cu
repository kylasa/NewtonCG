#include "gen_random.h"

#include "cuda_types.h"
#include "cuda_utils.h"

#include "time.h"

void getRandomVector (int n, real *hostPtr, real *devPtr) {

        curandGenerator_t gen ;
        int m = n + n % 2;

        /* Create pseudo - random number generator */
        curandCheckError ( curandCreateGenerator (&gen , CURAND_RNG_PSEUDO_DEFAULT ) );

        /* Set seed */
        //curandCheckError ( curandSetPseudoRandomGeneratorSeed ( gen , 1234ULL )) ;
        curandCheckError ( curandSetPseudoRandomGeneratorSeed ( gen , time(NULL) )) ;

        /* Generate n floats on device */
        //curandCheckError ( curandGenerateNormalDouble ( gen , devPtr , m, 0, 1.)) ;
        curandCheckError ( curandGenerateUniformDouble ( gen , devPtr , m)) ;

        /* Copy device memory to host */
        //copy_host_device( hostPtr, devPtr, sizeof(real) * n, cudaMemcpyDeviceToHost,
        //                        ERROR_MEMCPY_DEVICE_HOST );
        /* Cleanup */
        curandCheckError ( curandDestroyGenerator ( gen ) );
}

/*
Random Shuffle Here. 
https://stackoverflow.com/questions/15961119/how-to-create-a-random-permutation-of-an-array
*/
void randomShuffle( int *idx, int n)
{
	int j, temp; 
	for (int i = n - 1; i >= 0; i --){
		j = rand () % (i+1); 	

		temp = idx[i]; 
		idx[i] = idx[j]; 
		idx[j] = temp;
	}
}


/*
Floyd's algorithm Here. 
https://stackoverflow.com/questions/1608181/unique-random-numbers-in-an-integer-array-in-the-c-programming-language
*/

void genRandomVector( int *idx, int m, int n ) {

	int in, im; 
	int rn, rm; 	
	im = 0; 

	for (in = 0; in < n && im < m; ++in ){
		rn = n - in; 
		rm = m - im; 

		if (rand () % rn < rm ){
			idx[ im ++] = in + 1; 
		}
	}

	if ( im != m ){
		fprintf( stderr, "Failed to generate required number of random numbers ... "); 
		exit (-1); 
	}

	randomShuffle( idx, m ); 
}

