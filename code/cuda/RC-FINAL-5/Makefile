CC	 	= g++
NVCC	 	= nvcc
LIBS    	= -L/usr/local/cuda-8.0/lib64 -lm -lz -lcuda -lcudart -lcublas -lcusparse -lcurand -lpthread -m64
NVCCFLAGS  	= -I. -I/usr/local/cuda-8.0/include -arch=sm_60
CFLAGS  	= -I. -I/usr/local/cuda-8.0/include -Wall -funroll-loops -fstrict-aliasing -O3
DEBUGFLAGS	= -D__debug__ -D__STATISTICS__

DEFS    	= $(CFLAGS) $(DEBUGFLAGS)
NVCCDEFS 	= $(NVCCFLAGS) $(DEBUGFLAGS)
FLAG    	= $(DEFS) $(INCS) $(LIBS)
NVCCFLAG 	= $(NVCCDEFS) $(LIBS)

OBJ		=	utils.o dataset.o cuda_utils.o logistic_fn_indicator.o \
			mat_functions.o cuda_environment.o linesearch.o \
			conjugate_gradient.o newton_cg.o newton-driver.o print_utils.o \
			softmax_multiclass.o gen_random.o sparse_dataset.o \
			subsampling_helpers.o classification_kernels.o

all:   beta
beta: $(OBJ) Makefile
	$(NVCC) $(OBJ) -o NewtonCGSolver $(NVCCFLAG)

utils.o: utils.h
	$(CC) $(DEFS) -c utils.c

dataset.o: dataset.h
	$(CC) $(DEFS) -c dataset.c

cuda_utils.o: cuda_utils.h
	$(CC) $(DEFS) -c cuda_utils.c

logistic_fn_indicator.o: logistic_fn_indicator.h
	$(NVCC) $(NVCCDEFS) -c logistic_fn_indicator.cu

mat_functions.o: mat_functions.h
	$(NVCC) $(NVCCDEFS) -c mat_functions.cu

cuda_environment.o: cuda_environment.h
	$(CC) $(DEFS) -c cuda_environment.c

linesearch.o: linesearch.h
	$(CC) $(DEFS) -c linesearch.c

conjugate_gradient.o: conjugate_gradient.h
	$(CC) $(DEFS) -c conjugate_gradient.c

newton_cg.o: newton_cg.h
	$(CC) $(DEFS) -c newton_cg.c

newton-driver.o:
	$(CC) $(DEFS) -c newton-driver.c

print_utils.o:
	$(CC) $(DEFS) -c print_utils.c

softmax_multiclass.o: softmax_multiclass.h
	$(NVCC) $(NVCCDEFS) -c softmax_multiclass.cu

sparse_dataset.o: sparse_dataset.h
	$(NVCC) $(NVCCDEFS) -c sparse_dataset.cu

gen_random.o: gen_random.h
	$(NVCC) $(NVCCDEFS) -c gen_random.cu

subsampling_helpers.o: subsampling_helpers.h
	$(NVCC) $(NVCCDEFS) -c subsampling_helpers.cu

classification_kernels.o:  classification_kernels.h
	$(NVCC) $(NVCCDEFS) -c classification_kernels.cu

clean:
	rm -f *.o *~ core 
	rm NewtonCGSolver
