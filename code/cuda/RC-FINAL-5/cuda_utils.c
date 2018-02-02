
#include "cuda_utils.h"
#include "cuda_types.h"

void cuda_malloc (void **ptr, unsigned int size, int memset, int err_code) {

    cudaError_t retVal = cudaSuccess;
    retVal = cudaMalloc (ptr, size);
    if (retVal != cudaSuccess) {
		fprintf (stderr, "Failed to allocate memory on device for the res: %d...  exiting with code: %d size: %d, %s \n", 
							err_code, retVal, size, cudaGetErrorString(retVal));
		exit (err_code);
    }  

    if (memset) {
        retVal = cudaMemset (*ptr, 0, size);
        if (retVal != cudaSuccess) {
			fprintf (stderr, "Failed to memset memory on device... exiting with code %d, %s\n", 
							err_code, cudaGetErrorString( retVal ));
			exit (err_code);
        }
    }  
}

void cuda_malloc_host (void **ptr, unsigned int size, int memset, int err_code) {

    cudaError_t retVal = cudaSuccess;
    retVal = cudaMallocHost (ptr, size);
    if (retVal != cudaSuccess) {
		fprintf (stderr, "Failed to allocate memory on device for the res: %d...  exiting with code: %d size: %d, %s \n", 
							err_code, retVal, size, cudaGetErrorString(retVal) );
		exit (err_code);
    }  

    if (memset) {
        retVal = cudaMemset (*ptr, 0, size);
        if (retVal != cudaSuccess) {
			fprintf (stderr, "Failed to memset memory on device... exiting with code %d, %s\n", 
							err_code, cudaGetErrorString( retVal ));
			exit (err_code);
        }
    }  
}



void cuda_free (void *ptr, int err_code) {

    cudaError_t retVal = cudaSuccess;
    if (!ptr) return;

    retVal = cudaFree (ptr);

    if (retVal != cudaSuccess) {
		fprintf (stderr, "Failed to release memory on device for res %d... exiting with code %d -- Address %ld, %s\n", 
						err_code, retVal, (long int)ptr, cudaGetErrorString( retVal ));
        return;
    }  
}

void cuda_free_host (void *ptr, int err_code) {

    cudaError_t retVal = cudaSuccess;
    if (!ptr) return;

    retVal = cudaFreeHost (ptr);

    if (retVal != cudaSuccess) {
		fprintf (stderr, "Failed to release memory on device for res %d... exiting with code %d -- Address %ld, %s\n", 
						err_code, retVal, (long int)ptr, cudaGetErrorString( retVal ));
        return;
    }  
}


void cuda_memset (void *ptr, int data, size_t count, int err_code){
    cudaError_t retVal = cudaSuccess;

    retVal = cudaMemset (ptr, data, count);
    if (retVal != cudaSuccess) {
	 	fprintf (stderr, "ptr passed is %ld, value: %ld \n", (long int)ptr, &ptr);
	 	fprintf (stderr, " size to memset: %ld \n", count);
		fprintf (stderr, " target data is : %d \n", data);
		fprintf (stderr, "Failed to memset memory on device... exiting with code %d, cuda code %d, %s\n", 
							err_code, retVal, cudaGetErrorString( retVal ));
		exit (err_code);
    }
}

void copy_host_device (void *host, void *dev, int size, enum cudaMemcpyKind dir, int resid)
{
	cudaError_t	retVal = cudaErrorNotReady;

	if (dir == cudaMemcpyHostToDevice)
		retVal = cudaMemcpy (dev, host, size, cudaMemcpyHostToDevice);
	else
		retVal = cudaMemcpy (host, dev, size, cudaMemcpyDeviceToHost);

	if (retVal != cudaSuccess) {
		fprintf (stderr, "could not copy resource %d from host to device: reason %d:%s \n",
							resid, retVal, cudaGetErrorString( retVal ));
		exit (resid);
	}
}

void copy_device (void *dest, void *src, int size, int resid)
{
	cudaError_t	retVal = cudaErrorNotReady;

	retVal = cudaMemcpy (dest, src, size, cudaMemcpyDeviceToDevice);
	if (retVal != cudaSuccess) {
		fprintf (stderr, "could not copy resource %d from host to device: reason %d \n",
							resid, retVal);
		exit (resid);
	}
}

void print_device_mem_usage ()
{
   size_t total, free;
   cudaMemGetInfo (&free, &total);
   if (cudaGetLastError () != cudaSuccess )
   {
      fprintf (stderr, "Error on the memory call \n");
		return;
   }

   fprintf (stderr, "Total %ld Mb %ld gig %ld , free %ld, Mb %ld , gig %ld \n", 
                     total, total/(1024*1024), total/ (1024*1024*1024), 
                     free, free/(1024*1024), free/ (1024*1024*1024) );
}

void compute_blocks ( int *blocks, int *block_size, int count )
{
        *block_size = CUDA_BLOCK_SIZE;
        *blocks = (count / CUDA_BLOCK_SIZE ) + (count % CUDA_BLOCK_SIZE == 0 ? 0 : 1);
}

void compute_nearest_pow_2 (int blocks, int *result)
{
        int power = 1;
        while (power < blocks) power *= 2;

        *result = power;
}
