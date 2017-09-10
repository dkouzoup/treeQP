
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
    struct d_strmat *M;  // MAX(nx, nu) x MAX(nx, nu) matrix to store intermediate results
} stage_QP;

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TREE_H_ */
