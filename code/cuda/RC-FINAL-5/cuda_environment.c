#include <cuda_types.h>
#include <cuda_utils.h>
#include <utils.h>

#include <cuda_environment.h>

#include <stdlib.h>
#include <time.h>

void cuda_env_init(SCRATCH_AREA *scratch, int gpu){
	//cudaSetDevice (0);
	cudaSetDevice (gpu);
	cudaCheckError ();

        cudaDeviceReset ();
        cudaDeviceSynchronize ();

	allocate_memory( (void **)&scratch->hostWorkspace, (size_t)HOST_WORKSPACE_SIZE );
	cuda_malloc( (void **)&scratch->devWorkspace, DEVICE_WORKSPACE_SIZE, 1, ERR_MEM_ALLOC  );
	cuda_malloc_host ((void **)&scratch->pageLckWorkspace, PAGE_LOCKED_WORKSPACE_SIZE, 0, ERR_MEM_ALLOC );

	cublasCheckError( cublasCreate( &cublasHandle ) );
	cusparseCheckError( cusparseCreate( &cusparseHandle ) );

	allocate_memory( (void **)&dscratch, (size_t)DEBUG_SCRATCH_SIZE);

	srand( time(NULL) ); 
}

void cuda_env_cleanup (SCRATCH_AREA *scratch){
	release_memory( (void **)&scratch->hostWorkspace );
	cuda_free ((void *)scratch->devWorkspace, ERR_MEM_FREE);
	cuda_free_host ( (void *)scratch->pageLckWorkspace, ERR_MEM_FREE );

	release_memory( (void **)&dscratch);
}
