#ifndef __H_READ_MATRIX__
#define __H_READ_MATRIX__

#include <stdio.h>
#include <stdlib.h>
#include "ctype.h"

void tokenize( char *, double *, int* );
void readMatVec( char *, char *, double **, double **, int *);
void readVec( char *, double **, int *);


#endif
