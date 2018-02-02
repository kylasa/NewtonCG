#include <utils.h>
#include <sys/time.h>

void allocate_memory( void **ptr, size_t s )
{
	*ptr = malloc( s );
	if (*ptr == NULL){
		fprintf( stderr, "Memory Allocation failed for size: %u\n", s );
	}
}

void release_memory( void **ptr ){
	free ( *ptr );
}

real Get_Time( )
{
  struct timeval tim;
  
  gettimeofday(&tim, NULL );
  return( tim.tv_sec + (tim.tv_usec / 1000000.0) );
}


real Get_Timing_Info( real t_start )
{
  struct timeval tim;
  real t_end;
  
  gettimeofday(&tim, NULL );
  t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
  return (t_end - t_start);
}
