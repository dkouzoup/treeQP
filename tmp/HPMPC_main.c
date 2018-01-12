#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_common.h"
#include "blasfeo/include/blasfeo_v_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_i_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_blas.h"

#include "hpmpc/include/target.h"
#include "hpmpc/include/tree.h"
#include "hpmpc/include/lqcp_solvers.h"
#include "hpmpc/include/mpc_solvers.h"

#include "HPMPC_data.c"

#define TIC_TOC

#ifdef TIC_TOC
#include "treeqp/utils/timing.h"
#endif

#include "treeqp/src/tree_ocp_qp_common.h"
#include "treeqp/src/hpmpc_tree.h"
#include "treeqp/utils/utils.h"
#include "treeqp/utils/tree_utils.h"


// choose the solution algorithm
// 0: all
// 1: standard Riccati for nominal MPC
// 2: tree Riccati for tree MPC
// 3: standard Riccati for tree MPC

// printint
#define PRINT


int main()
	{

	int ii, jj;
    int niter;

	struct blasfeo_dmat *hsmatdummy;
	struct blasfeo_dvec *hsvecdummy;

#ifdef TIC_TOC
	treeqp_timer tot_tmr;
	double total_time;
#endif

/************************************************
* IPM arguments
************************************************/

	// IPM args
	treeqp_hpmpc_options_t opts = treeqp_hpmpc_default_options();
	int hpmpc_status;

	// timing
	struct timeval tv0, tv1;
#ifdef TIC_TOC
    int nrep = 10;
#else
	int nrep = 100;
#endif
	int rep;

/************************************************
*	setup problem data
************************************************/

#if defined(PRINT)
	printf("\nnx = %d, nu = %d\n\n", nx_, nu_);
#endif

	// initial state
	struct blasfeo_dvec sx0; blasfeo_allocate_dvec(nx_, &sx0);
#if 1
	int c;
	double x0[nx_];
	FILE *file_x0 = fopen("HPMPC_x0.txt", "r");
	if(file_x0)
		{
		for(ii=0; ii<nx_; ii++)
			{
			c = fscanf(file_x0, "%le", &x0[ii]);
			}
		fclose(file_x0);
		}
	else
		{
		printf("\nx0 file not found\n");
		exit(1);
		}
#endif
	blasfeo_pack_dvec(nx_, x0, &sx0, 0);
#if defined(PRINT)
	blasfeo_print_tran_dvec(nx_, &sx0, 0);
#endif

	// dynamic
	struct blasfeo_dmat sA; blasfeo_allocate_dmat(nx_, nx_, &sA);

	// 0 nominal + md realizations 1 to md
	struct blasfeo_dvec sb0[md+1];
	struct blasfeo_dvec sb1[md+1];
	struct blasfeo_dmat sBbt0[md+1];
	struct blasfeo_dmat sBAbt1[md+1];
	for(ii=0; ii<md+1; ii++)
		{
		blasfeo_allocate_dvec(nx_, &sb0[ii]);
		blasfeo_allocate_dvec(nx_, &sb1[ii]);
		blasfeo_allocate_dmat(nu_+1, nx_, &sBbt0[ii]);
		blasfeo_allocate_dmat(nu_+nx_+1, nx_, &sBAbt1[ii]);
		}

	// stage 0
	for(ii=0; ii<md+1; ii++)
		{
		blasfeo_pack_dmat(nx_, nx_, A+ii*nx_*nx_, nx_, &sA, 0, 0);
//		blasfeo_print_dmat(nx_, nx_, &sA, 0, 0);
		blasfeo_pack_dvec(nx_, b+ii*nx_, &sb1[ii], 0);
		blasfeo_dgemv_n(nx_, nx_, 1.0, &sA, 0, 0, &sx0, 0, 1.0, &sb1[ii], 0, &sb0[ii], 0);
//		blasfeo_print_tran_dvec(nx_, &sb0[ii], 0);
		blasfeo_pack_tran_dmat(nx_, nu_, B+ii*nx_*nu_, nx_, &sBbt0[ii], 0, 0);
		blasfeo_drowin(nx_, 1.0, &sb0[ii], 0, &sBbt0[ii], nu_, 0);
#if defined(PRINT)
		blasfeo_print_dmat(nu_+1, nx_, &sBbt0[ii], 0, 0);
#endif
		}

	// stage 1
	for(ii=0; ii<md+1; ii++)
		{
		blasfeo_pack_dvec(nx_, b+ii*nx_, &sb1[ii], 0);
		blasfeo_pack_tran_dmat(nx_, nu_, B+ii*nx_*nu_, nx_, &sBAbt1[ii], 0, 0);
		blasfeo_pack_tran_dmat(nx_, nx_, A+ii*nx_*nx_, nx_, &sBAbt1[ii], nu_, 0);
		blasfeo_pack_tran_dmat(nx_, 1, b+ii*nx_, nx_, &sBAbt1[ii], nu_+nx_, 0);
#if defined(PRINT)
		blasfeo_print_dmat(nu_+nx_+1, nx_, &sBAbt1[ii], 0, 0);
#endif
		}

	// cost function matrices
	struct blasfeo_dvec sr; blasfeo_allocate_dvec(nu_, &sr);
	blasfeo_pack_dvec(nu_, r, &sr, 0);

	struct blasfeo_dvec sq; blasfeo_allocate_dvec(nx_, &sq);
	blasfeo_pack_dvec(nx_, q, &sq, 0);

	struct blasfeo_dvec sp; blasfeo_allocate_dvec(nx_, &sp);
	blasfeo_pack_dvec(nx_, p, &sp, 0);

	struct blasfeo_dvec sdR; blasfeo_allocate_dvec(nu_, &sdR);
	blasfeo_pack_dvec(nu_, dR, &sdR, 0);

	struct blasfeo_dvec sdQ; blasfeo_allocate_dvec(nx_, &sdQ);
	blasfeo_pack_dvec(nx_, dQ, &sdQ, 0);

	struct blasfeo_dvec sdP; blasfeo_allocate_dvec(nx_, &sdP);
	blasfeo_pack_dvec(nx_, dP, &sdP, 0);

	// pack cost function
	struct blasfeo_dvec sr0; blasfeo_allocate_dvec(nu_, &sr0);
	blasfeo_dveccp(nu_, &sr, 0, &sr0, 0); // XXX no need to update r0 since S=0

	struct blasfeo_dmat sRr0; blasfeo_allocate_dmat(nu_+1, nu_, &sRr0);
	blasfeo_ddiain(nu_, 1.0, &sdR, 0, &sRr0, 0, 0);
	blasfeo_drowin(nu_, 1.0, &sr0, 0, &sRr0, nu_, 0);
#if defined(PRINT)
	blasfeo_print_dmat(nu_+1, nu_, &sRr0, 0, 0);
#endif

	struct blasfeo_dvec srq1; blasfeo_allocate_dvec(nu_+nx_, &srq1);
	blasfeo_dveccp(nu_, &sr, 0, &srq1, 0);
	blasfeo_dveccp(nx_, &sq, 0, &srq1, nu_);

	struct blasfeo_dmat sRSQrq1; blasfeo_allocate_dmat(nu_+nx_+1, nu_+nx_, &sRSQrq1);
	blasfeo_ddiain(nu_, 1.0, &sdR, 0, &sRSQrq1, 0, 0);
	blasfeo_ddiain(nx_, 1.0, &sdQ, 0, &sRSQrq1, nu_, nu_);
	blasfeo_drowin(nu_, 1.0, &sr, 0, &sRSQrq1, nu_+nx_, 0);
	blasfeo_drowin(nx_, 1.0, &sq, 0, &sRSQrq1, nu_+nx_, nu_);
#if defined(PRINT)
	blasfeo_print_dmat(nu_+nx_+1, nu_+nx_, &sRSQrq1, 0, 0);
#endif

	struct blasfeo_dvec sqN; blasfeo_allocate_dvec(nx_, &sqN);
	blasfeo_dveccp(nx_, &sp, 0, &sqN, 0);

	struct blasfeo_dmat sQqN; blasfeo_allocate_dmat(nx_+1, nx_, &sQqN);
	blasfeo_ddiain(nx_, 1.0, &sdP, 0, &sQqN, 0, 0);
	blasfeo_drowin(nx_, 1.0, &sp, 0, &sQqN, nx_, 0);
#if defined(PRINT)
	blasfeo_print_dmat(nx_+1, nx_, &sQqN, 0, 0);
#endif

	// bounds
	struct blasfeo_dvec sd0; blasfeo_allocate_dvec(2*nu_, &sd0);
	blasfeo_pack_dvec(nu_, umin, &sd0, 0);
	blasfeo_pack_dvec(nu_, umax, &sd0, nu_);
#if defined(PRINT)
	blasfeo_print_dvec(2*nu_, &sd0, 0);
#endif
	int *idxb0; int_zeros(&idxb0, nu_, 1);
	for(ii=0; ii<nu_; ii++)
		idxb0[ii] = ii;

	struct blasfeo_dvec sd1; blasfeo_allocate_dvec(2*nu_+2*nx_, &sd1);
	blasfeo_pack_dvec(nu_, umin, &sd1, 0);
	blasfeo_pack_dvec(nu_, umax, &sd1, nu_+nx_);
	blasfeo_pack_dvec(nx_, xmin, &sd1, nu_);
	blasfeo_pack_dvec(nx_, xmax, &sd1, nu_+nx_+nu_);
#if defined(PRINT)
	blasfeo_print_dvec(2*nu_+2*nx_, &sd1, 0);
#endif
	int *idxb1; int_zeros(&idxb1, nu_+nx_, 1);
	for(ii=0; ii<nu_+nx_; ii++)
		idxb1[ii] = ii;

	struct blasfeo_dvec sdN; blasfeo_allocate_dvec(2*nx_, &sdN);
	blasfeo_pack_dvec(nx_, xmin, &sdN, 0);
	blasfeo_pack_dvec(nx_, xmax, &sdN, nx_);
#if defined(PRINT)
	blasfeo_print_dvec(2*nx_, &sdN, 0);
#endif
	int *idxbN; int_zeros(&idxbN, nx_, 1);
	for(ii=0; ii<nx_; ii++)
		idxbN[ii] = ii;

/************************************************
* no-tree MPC: array of matrices
************************************************/

	int N = Nh;

	int nx[N+1];
	nx[0] = 0;
	for(ii=1; ii<=N; ii++)
		nx[ii] = nx_;

	int nu[N+1];
	for(ii=0; ii<N; ii++)
		nu[ii] = nu_;
	nu[N] = 0;

	int nb[N+1];
	nb[0] = nu_;
	for(ii=1; ii<N; ii++)
		nb[ii] = nu_+nx_;
	nb[N] = nx_;

	int ng[N+1];
	for(ii=0; ii<=N; ii++)
		ng[ii] = 0;

#if defined(PRINT)
	for(ii=0; ii<=N; ii++)
		printf("\n%d %d %d %d\n", nx[ii], nu[ii], nb[ii], ng[ii]);
#endif

	// data & solution
	struct blasfeo_dmat hsBAbt[N];
	struct blasfeo_dvec hsb[N];
	struct blasfeo_dmat hsRSQrq[N+1];
	struct blasfeo_dvec hsrq[N+1];
	struct blasfeo_dvec hsdRSQ[N+1];
	struct blasfeo_dmat hsDCt[N+1];
	struct blasfeo_dvec hsd[N+1];
	struct blasfeo_dvec hsux[N+1];
	struct blasfeo_dvec hspi[N+1];
	struct blasfeo_dvec hslam[N+1];
	struct blasfeo_dvec hst[N+1];
	int *hidxb[N+1];
	// residuals
	struct blasfeo_dvec hsrrq[N+1];
	struct blasfeo_dvec hsrb[N];
	struct blasfeo_dvec hsrd[N+1];
	struct blasfeo_dvec hsrm[N+1];
	double mu;
	// work space
	struct blasfeo_dmat hsL[N+1];
	void *work_ipm;
	void *work_res;

	hsBAbt[0] = sBbt0[0];
	hsb[0] = sb0[0];
	hsRSQrq[0] = sRr0;
	hsrq[0] = sr0;
	hsd[0] = sd0;
	blasfeo_allocate_dvec(nu[0]+nx[0]+1, &hsux[0]);
	blasfeo_allocate_dvec(nx[0], &hspi[0]);
	blasfeo_allocate_dvec(2*nb[0]+2*ng[0], &hslam[0]);
	blasfeo_allocate_dvec(2*nb[0]+2*ng[0], &hst[0]);
	hidxb[0] = idxb0;
	blasfeo_allocate_dvec(nu[0]+nx[0], &hsrrq[0]);
	blasfeo_allocate_dvec(nx[1], &hsrb[0]);
	blasfeo_allocate_dvec(2*nb[0]+2*ng[0], &hsrd[0]);
	blasfeo_allocate_dvec(2*nb[0]+2*ng[0], &hsrm[0]);
	blasfeo_allocate_dmat(nu[0]+nx[0]+1, nu[0]+nx[0], &hsL[0]);
	for(ii=1; ii<N; ii++)
		{
		hsBAbt[ii] = sBAbt1[0];
		hsb[ii] = sb1[0];
		hsRSQrq[ii] = sRSQrq1;
		hsrq[ii] = srq1;
		hsd[ii] = sd1;
		blasfeo_allocate_dvec(nu[ii]+nx[ii]+1, &hsux[ii]);
		blasfeo_allocate_dvec(nx[ii], &hspi[ii]);
		blasfeo_allocate_dvec(2*nb[ii]+2*ng[ii], &hslam[ii]);
		blasfeo_allocate_dvec(2*nb[ii]+2*ng[ii], &hst[ii]);
		hidxb[ii] = idxb1;
		blasfeo_allocate_dvec(nu[ii]+nx[ii], &hsrrq[ii]);
		blasfeo_allocate_dvec(nx[ii+1], &hsrb[ii]);
		blasfeo_allocate_dvec(2*nb[ii]+2*ng[ii], &hsrd[ii]);
		blasfeo_allocate_dvec(2*nb[ii]+2*ng[ii], &hsrm[ii]);
		blasfeo_allocate_dmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], &hsL[ii]);
		}
	hsRSQrq[N] = sQqN;
	hsrq[N] = sqN;
	hsd[N] = sdN;
	blasfeo_allocate_dvec(nu[N]+nx[N]+1, &hsux[N]);
	blasfeo_allocate_dvec(nx[N], &hspi[N]);
	blasfeo_allocate_dvec(2*nb[N]+2*ng[N], &hslam[N]);
	blasfeo_allocate_dvec(2*nb[N]+2*ng[N], &hst[N]);
	hidxb[N] = idxbN;
	blasfeo_allocate_dvec(nu[N]+nx[N], &hsrrq[N]);
	blasfeo_allocate_dvec(2*nb[N]+2*ng[N], &hsrd[N]);
	blasfeo_allocate_dvec(2*nb[N]+2*ng[N], &hsrm[N]);
	blasfeo_allocate_dmat(nu[N]+nx[N]+1, nu[N]+nx[N], &hsL[N]);

	v_zeros_align(&work_ipm, d_ip2_res_mpc_hard_work_space_size_bytes_libstr(N, nx, nu, nb, ng));
	v_zeros_align(&work_res, d_res_res_mpc_hard_work_space_size_bytes_libstr(N, nx, nu, nb, ng));


/************************************************
* scenario-tree MPC with tailroed solver
************************************************/

	int Nn = calculate_number_of_nodes(md, Nr, Nh);

#if defined(PRINT)
	printf("\nnumber of nodes = %d\n", Nn);
#endif

	int nkids, idxkid;

	// struct node tree[Nn];
	struct node *tree = malloc(Nn*sizeof(struct node));

	// setup the tree
	setup_multistage_tree(md, Nr, Nh, Nn, tree);

#if defined(PRINT)
	// // print the tree
	// for(ii=0; ii<Nn; ii++)
	// 	print_node(&tree[ii]);
#endif

	// data structure

	int stage, real;

	// stage-wise variant size (tmp)
	int *t_nx = malloc(Nn*sizeof(int));
	int *t_nu = malloc(Nn*sizeof(int));
	int *t_nb = malloc(Nn*sizeof(int));
	int *t_ng = malloc(Nn*sizeof(int));

	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage;
		t_nx[ii] = nx[stage];
		t_nu[ii] = nu[stage];
		t_nb[ii] = nb[stage];
		t_ng[ii] = ng[stage];
		}

	// dynamics indexed by node (ecluding the root) // no real atm
	struct blasfeo_dmat t_hsBAbt[Nn-1]; // XXX defined on the edges !!!!!
	struct blasfeo_dvec t_hsb[Nn-1]; // XXX defined on the edges !!!!!
	for(ii=0; ii<Nn-1; ii++)
		{
		stage = tree[ii+1].stage;
		real = tree[ii+1].real+1; // XXX realization 0 is the nominal case !!!!!
		if(stage==1)
			{
			t_hsBAbt[ii] = sBbt0[real];
			t_hsb[ii] = sb0[real];
			}
		else
			{
			t_hsBAbt[ii] = sBAbt1[real];
			t_hsb[ii] = sb1[real];
			}
		}
#if 0
	for(ii=0; ii<Nn-1; ii++)
		{
		stage = tree[ii+1].stage;
		real = tree[ii+1].real;
		printf("\nstage = %d, real = %d\n", stage, real);
		blasfeo_print_dmat(t_nu[stage-1]+t_nx[stage-1]+1, t_nx[stage], &t_hsBAbt[ii], 0, 0);
		}
	return 0;
#endif

	// temporary cost function indexed by stage
	struct blasfeo_dmat tmp_hsRSQrq[Nh+1];
	struct blasfeo_dvec tmp_hsrq[Nh+1];
	// first stages: scale cost function
	for(ii=0; ii<Nr; ii++)
		{
		blasfeo_allocate_dmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], &tmp_hsRSQrq[ii]);
		blasfeo_allocate_dvec(nu[ii]+nx[ii], &tmp_hsrq[ii]);
		}
	// last stages: original cost function
	for(ii=Nr; ii<Nh; ii++)
		{
		tmp_hsRSQrq[ii] = sRSQrq1;
		tmp_hsrq[ii] = srq1;
		}
	// last stage
	tmp_hsRSQrq[Nh] = sQqN;
	tmp_hsrq[Nh] = sqN;
	// scale at first stages
	for(ii=Nr-1; ii>0; ii--)
		{
		blasfeo_dgecp(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], &tmp_hsRSQrq[ii+1], 0, 0, &tmp_hsRSQrq[ii], 0, 0);
		blasfeo_dgesc(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], md, &tmp_hsRSQrq[ii], 0, 0);
		blasfeo_dveccp(nu[ii]+nx[ii], &tmp_hsrq[ii+1], 0, &tmp_hsrq[ii], 0);
		blasfeo_dvecsc(nu[ii]+nx[ii], md, &tmp_hsrq[ii], 0);
		}
	// scale first stage
	ii = 0;
	blasfeo_dgecp(nu[ii]+nx[ii], nu[ii]+nx[ii], &tmp_hsRSQrq[ii+1], 0, 0, &tmp_hsRSQrq[ii], 0, 0);
	blasfeo_dgesc(nu[ii]+nx[ii], nu[ii]+nx[ii], md, &tmp_hsRSQrq[ii], 0, 0);
	blasfeo_dgecp(1, nu[ii]+nx[ii], &tmp_hsRSQrq[ii+1], nu[ii+1]+nx[ii+1], 0, &tmp_hsRSQrq[ii], nu[ii]+nx[ii], 0);
	blasfeo_dgesc(1, nu[ii]+nx[ii], md, &tmp_hsRSQrq[ii], nu[ii]+nx[ii], 0);
	blasfeo_dveccp(nu[ii]+nx[ii], &tmp_hsrq[ii+1], 0, &tmp_hsrq[ii], 0);
	blasfeo_dvecsc(nu[ii]+nx[ii], md, &tmp_hsrq[ii], 0);

//	for(ii=0; ii<=Nh; ii++)
//		{
//		blasfeo_print_dmat(t_nu[ii]+t_nx[ii]+1, t_nu[ii]+t_nx[ii], &tmp_hsRSQrq[ii], 0, 0);
//		}
//	return;

	// cost function indexed by node
	// struct blasfeo_dmat t_hsRSQrq[Nn];
	// struct blasfeo_dvec t_hsrq[Nn];
	struct blasfeo_dmat *t_hsRSQrq = malloc(Nn*sizeof(struct blasfeo_dmat));
	struct blasfeo_dvec *t_hsrq = malloc(Nn*sizeof(struct blasfeo_dvec));
	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage;
		t_hsRSQrq[ii] = tmp_hsRSQrq[stage];
		t_hsrq[ii] = tmp_hsrq[stage];
		}

	// constraints indexed by node
	// struct blasfeo_dmat t_hsDCt[Nn];
	// struct blasfeo_dvec t_hsd[Nn];
	int *t_hidxb[Nn];

	struct blasfeo_dmat *t_hsDCt = malloc(Nn*sizeof(struct blasfeo_dmat));
	struct blasfeo_dvec *t_hsd = malloc(Nn*sizeof(struct blasfeo_dvec));

	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage;
		t_hsd[ii] = hsd[stage];
		t_hidxb[ii] = hidxb[stage];
		}

	// res
	struct blasfeo_dvec *t_hsrrq = malloc(Nn*sizeof(struct blasfeo_dvec));
	struct blasfeo_dvec *t_hsrb = malloc((Nn-1)*sizeof(struct blasfeo_dvec));
	struct blasfeo_dvec *t_hsrd = malloc(Nn*sizeof(struct blasfeo_dvec));
	struct blasfeo_dvec *t_hsrm = malloc(Nn*sizeof(struct blasfeo_dvec));

	// work
	struct blasfeo_dmat *t_hsL = malloc(Nn*sizeof(struct blasfeo_dmat));
	void *t_work_res;
	for(ii=0; ii<Nn; ii++)
		{
		blasfeo_allocate_dvec(t_nu[ii]+t_nx[ii], &t_hsrrq[ii]);
		blasfeo_allocate_dvec(2*t_nb[ii]+2*t_ng[ii], &t_hsrd[ii]);
		blasfeo_allocate_dvec(2*t_nb[ii]+2*t_ng[ii], &t_hsrm[ii]);
		blasfeo_allocate_dmat(t_nu[ii]+t_nx[ii]+1, t_nu[ii]+t_nx[ii], &t_hsL[ii]);
		for(jj=0; jj<tree[ii].nkids; jj++)
			{
			idxkid = tree[ii].kids[jj];
			blasfeo_allocate_dvec(t_nx[idxkid], &t_hsrb[idxkid-1]);
			}
		}

	v_zeros_align(&t_work_res, d_tree_res_res_mpc_hard_work_space_size_bytes_libstr(Nn, tree, t_nx, t_nu, t_nb, t_ng));

	// IPM work space
#if defined(PRINT)
	printf("\ntree work space size bytes %d\n\n", d_ip2_res_mpc_hard_work_space_size_bytes_libstr(Nn-1, t_nx, t_nu, t_nb, t_ng));
#endif


#if defined(PRINT)
	printf("\nsolving tree MPC with tree HPMPC ...\n\n");
#endif


// *****************************************************************************

// setup QP
tree_ocp_qp_in qp_in;

int_t qp_in_size = tree_ocp_qp_in_calculate_size(Nn, t_nx, t_nu, tree);
void *qp_in_memory = malloc(qp_in_size);
create_tree_ocp_qp_in(Nn, t_nx, t_nu, tree, &qp_in, qp_in_memory);

// NOTE(dimitris): skipping first dynamics that represent the nominal ones
tree_ocp_qp_in_fill_lti_data_diag_weights(&A[nx_*nx_], &B[nx_*nu_], &b[nx_], dQ, q, dP, p, dR, r,
	xmin, xmax, umin, umax, x0, &qp_in);

// setup QP solver
treeqp_hpmpc_workspace work;

int_t treeqp_size = treeqp_hpmpc_calculate_size(&qp_in, &opts);
void *qp_solver_memory = malloc(treeqp_size);
create_treeqp_hpmpc(&qp_in, &opts, &work, qp_solver_memory);

// setup QP solution
tree_ocp_qp_out qp_out;

int_t qp_out_size = tree_ocp_qp_out_calculate_size(Nn, t_nx, t_nu);
void *qp_out_memory = malloc(qp_out_size);
create_tree_ocp_qp_out(Nn, t_nx, t_nu, &qp_out, qp_out_memory);



// *****************************************************************************

#if 0
for(ii=0; ii<N; ii++)
	blasfeo_print_dmat(nu[ii]+nx[ii]+1, nx[ii+1], &t_hsBAbt[ii], 0, 0);
#endif

	// call ipm
#ifdef TIC_TOC
	treeqp_tic(&tot_tmr);
#else
	gettimeofday(&tv0, NULL); // time
#endif

	for (rep = 0; rep < nrep; rep++) {
		treeqp_hpmpc_solve(&qp_in, &qp_out, &opts, &work);
	}

#ifdef TIC_TOC
	total_time = treeqp_toc(&tot_tmr);
#else
	gettimeofday(&tv1, NULL); // time
#endif
    niter = qp_out.info.iter;

#ifdef TIC_TOC
	float time_tree_ipm = (float) total_time/nrep;
#else
	float time_tree_ipm = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
#endif

// *****************************************************************************

// copy results to qp_out struct
for (ii = 0; ii < Nn; ii++) {
	blasfeo_dveccp(qp_in.nu[ii], &work.sux[ii], 0, &qp_out.u[ii], 0);
	blasfeo_dveccp(qp_in.nx[ii], &work.sux[ii], qp_in.nu[ii], &qp_out.x[ii], 0);
}

// *****************************************************************************


#if defined(PRINT)
	printf("\n... done\n\n");

	// print sol
	printf("\nt_ux = \n\n");
	for(ii=0; ii<Nn; ii++)
		{
		blasfeo_print_tran_dvec(t_nu[ii]+t_nx[ii], &work.sux[ii], 0);
		}
	printf("\nt_pi = \n\n");
	for(ii=0; ii<Nn; ii++)
		{
		blasfeo_print_tran_dvec(t_nx[ii], &qp_out.lam[ii], 0);
		}

	// // print res
	// printf("\nt_res_rq\n");
	// for(ii=0; ii<Nn; ii++)
	// 	blasfeo_print_exp_tran_dvec(t_nu[ii]+t_nx[ii], &t_hsrrq[ii], 0);

	// printf("\nt_res_b\n");
	// for(ii=0; ii<Nn-1; ii++)
	// 	blasfeo_print_exp_tran_dvec(t_nx[ii+1], &t_hsrb[ii], 0);

	// printf("\nt_res_d\n");
	// for(ii=0; ii<Nn; ii++)
	// 	blasfeo_print_exp_tran_dvec(2*t_nb[ii]+2*t_ng[ii], &t_hsrd[ii], 0);

	// printf("\nt_res_m\n");
	// for(ii=0; ii<Nn; ii++)
	// 	blasfeo_print_exp_tran_dvec(2*t_nb[ii]+2*t_ng[ii], &t_hsrm[ii], 0);

	printf("\nstatistics from last run\n\n");
	for(jj=0; jj<qp_out.info.iter; jj++)
		printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\talpha = %f\tmu = %f\tmu = %e\n", jj, work.status[5*jj], work.status[5*jj+1], work.status[5*jj+2], work.status[5*jj+2], work.status[5*jj+3], work.status[5*jj+4], work.status[5*jj+4]);
	printf("\n");

	printf("\ntime tree ipm\n");
	printf("\n%e\n", time_tree_ipm);
	printf("\n");

	// print last element of sol
	printf("\nt_ux[Nn] = \n\n");
	blasfeo_print_tran_dvec(t_nu[Nn-1]+t_nx[Nn-1], &work.sux[Nn-1], 0);
#endif


/************************************************
*	print resutls to file
************************************************/

	FILE *tfile = fopen("tHPMPC.txt", "w");
	FILE *xfile = fopen("xHPMPC.txt", "w");
	FILE *ifile = fopen("iHPMPC.txt", "w");

	fprintf(tfile, "%1.15f\n", time_tree_ipm);
	fprintf(ifile, "%d\n", niter);
	for(jj=0; jj<nx_; jj++)
		fprintf(xfile, "%1.15f\n", blasfeo_dvecex1(&sx0, jj));
	for(ii=0; ii<Nn; ii++) // loop over nodes
		{
		for(jj=0; jj<t_nx[ii]; jj++)
			fprintf(xfile, "%1.15f\n", blasfeo_dvecex1(&work.sux[ii], t_nu[ii]+jj));
		for(jj=0; jj<t_nu[ii]; jj++)
			fprintf(xfile, "%1.15f\n", blasfeo_dvecex1(&work.sux[ii], jj));
		}

	fclose(tfile);
	fclose(xfile);
	fclose(ifile);

/************************************************
*	free memory
************************************************/

	// IPM common

	// nominal MPC
	blasfeo_free_dvec(&sx0);
	blasfeo_free_dmat(&sA);
	blasfeo_free_dvec(&sr);
	blasfeo_free_dvec(&sq);
	blasfeo_free_dvec(&sp);
	blasfeo_free_dvec(&sdR);
	blasfeo_free_dvec(&sdQ);
	blasfeo_free_dvec(&sdP);
	blasfeo_free_dvec(&sr0);
	for(ii=0; ii<md+1; ii++)
		{
		blasfeo_free_dvec(&sb0[ii]);
		blasfeo_free_dvec(&sb1[ii]);
		blasfeo_free_dmat(&sBbt0[ii]);
		blasfeo_free_dmat(&sBAbt1[ii]);
		}
	for(ii=0; ii<Nr; ii++)
		{
		blasfeo_free_dmat(&tmp_hsRSQrq[ii]);
		blasfeo_free_dvec(&tmp_hsrq[ii]);
		}
	blasfeo_free_dmat(&sRr0);
	blasfeo_free_dvec(&srq1);
	blasfeo_free_dmat(&sRSQrq1);
	blasfeo_free_dvec(&sqN);
	blasfeo_free_dmat(&sQqN);
	blasfeo_free_dvec(&sd0);
	blasfeo_free_dvec(&sd1);
	blasfeo_free_dvec(&sdN);
	int_free(idxb0);
	int_free(idxb1);
	int_free(idxbN);
	d_free_align(work_ipm);
	d_free_align(work_res);
	for(ii=0; ii<N; ii++)
		{
		blasfeo_free_dvec(&hsux[ii]);
		blasfeo_free_dvec(&hspi[ii]);
		blasfeo_free_dvec(&hslam[ii]);
		blasfeo_free_dvec(&hst[ii]);
		blasfeo_free_dvec(&hsrrq[ii]);
		blasfeo_free_dvec(&hsrb[ii]);
		blasfeo_free_dvec(&hsrd[ii]);
		blasfeo_free_dvec(&hsrm[ii]);
		blasfeo_free_dmat(&hsL[ii]);
		}
	ii = N;
	blasfeo_free_dvec(&hsux[ii]);
	blasfeo_free_dvec(&hspi[ii]);
	blasfeo_free_dvec(&hslam[ii]);
	blasfeo_free_dvec(&hst[ii]);
	blasfeo_free_dvec(&hsrrq[ii]);
	blasfeo_free_dvec(&hsrd[ii]);
	blasfeo_free_dvec(&hsrm[ii]);
	blasfeo_free_dmat(&hsL[ii]);

	// tree MPC with tree HPMPC
	d_free_align(t_work_res);
	for(ii=0; ii<Nn; ii++)
		{
		blasfeo_free_dmat(&t_hsL[ii]);
		}
	free_tree(Nn, tree);


    free(tree);

	free(t_nx);
	free(t_nu);
	free(t_nb);
	free(t_ng);

	free(t_hsRSQrq);
	free(t_hsrq);

	free(t_hsDCt);
	free(t_hsd);

	free(t_hsrrq);
	free(t_hsrb);
	free(t_hsrd);
	free(t_hsrm);

	free(t_hsL);

	return 0;

	}
