
#ifndef TREE_H_
#define TREE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "treeqp/flags.h"
#include "treeqp/utils/types.h"

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"

typedef struct stage_QP_ {

    #ifdef _CHECK_LAST_ACTIVE_SET_
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

// TODO(dimitris): outdated?
// NOTE(dimitris):
// - There are as many Hessian blocks on the diagonal as parent nodes in the tree
// - Each of those blocks except for the root has a lower diagonal block from its parent
// - The upper diagonal blocks from the children are neglected due to symmetry
// - Dimension of block k is (nc[k]*nx) x (nc[k]*nx), where nc[k] = tree[k].nkids
// - Dimension of parent block is (nc[k]*nx) x nx

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREE_H_ */
