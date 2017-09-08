
#ifndef TREE_H_
#define TREE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/flags.h"
#include "treeqp/utils/types.h"

// #include "acados/utils/types.h"
#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"

typedef struct stage_QP_ {

    // objective
    struct d_strvec *Q;
    struct d_strvec *R;
    struct d_strvec *q;
    struct d_strvec *r;
    struct d_strvec *Qinv;
    struct d_strvec *Rinv;

    // dynamics
    struct d_strmat *A;
    struct d_strmat *B;
    struct d_strvec *b;

    // bounds
    struct d_strvec *xmin;
    struct d_strvec *xmax;
    struct d_strvec *umin;
    struct d_strvec *umax;

    // iterates
    struct d_strvec *x;
    struct d_strvec *u;

    // intermediate results
    struct d_strvec *qmod;
    struct d_strvec *rmod;
    struct d_strvec *xas;
    struct d_strvec *uas;
    #ifdef _CHECK_LAST_ACTIVE_SET_
    struct d_strvec *xasPrev;
    struct d_strvec *uasPrev;
    struct d_strmat *Wdiag;  // diagonal nx x nx block of dual Hessian that corresponds to node
    int_t xasChanged;
    int_t uasChanged;
    #endif
    struct d_strvec *QinvCal;
    struct d_strvec *RinvCal;
    struct d_strmat *M;  // MAX(nx, nu) x MAX(nx, nu) matrix to store intermediate results
    real_t fval;
    real_t cmod;

} stage_QP;

// Note(dimitris):
// - There are as many Hessian blocks on the diagonal as parent nodes in the tree
// - Each of those blocks except for the root has a lower diagonal block from its parent
// - The upper diagonal blocks from the children are neglected due to symmetry
// - Dimension of block k is (nc[k]*nx) x (nc[k]*nx), where nc[k] = tree[k].nkids
// - Dimension of parent block is (nc[k]*nx) x nx

typedef struct dual_block_ {

    struct d_strvec *lambda;
    struct d_strvec *deltalambda;

} dual_block;


// Options of QP solver
typedef struct
{
	// iterations
	int_t maxIter;
    int_t lineSearchMaxIter;

    // numerical tolerances
	real_t stationarityTolerance;

    // termination condition options
    termination_t termCondition;

    // regularization options
    regType_t regType;
    real_t regValue;

    // line search options
    real_t lineSearchGamma;
    real_t lineSearchBeta;

} tree_options_t;

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREE_H_ */
