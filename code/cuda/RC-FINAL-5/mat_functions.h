#ifndef __H_MAT_FUNCTIONS__
#define __H_MAT_FUNCTIONS__

#include "cuda_types.h"

GLOBAL void ker_log_sum( real *t, real *target, int N, real *out);
GLOBAL void ker_sigmoid( real *target, int N, real *out);
GLOBAL void ker_sigmoid_classify( real *target, int N );
GLOBAL void ker_sigmoid_target( real *t, real *target, int N, real *out);
GLOBAL void ker_ele_vec_product( real *t1, real *t2, int N, real *out);
GLOBAL void ker_mat_identity (real *h, real reg_term, int M);
GLOBAL void ker_hx_matvec_reg ( real *hx, real gamma, real *vec, int c);
GLOBAL void ker_reduction(const real *h, real *out, int dim);

#endif
