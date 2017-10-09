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

	struct d_strmat *hsmatdummy;
	struct d_strvec *hsvecdummy;

#ifdef TIC_TOC
	treeqp_timer tot_tmr;
	double total_time;
#endif

/************************************************
* IPM arguments
************************************************/

	// IPM args
	treeqp_hpmpc_options_t opts;
	treeqp_hpmpc_set_default_options(&opts);

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
	struct d_strvec sx0; d_allocate_strvec(nx_, &sx0);
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
	d_cvt_vec2strvec(nx_, x0, &sx0, 0);
#if defined(PRINT)
	d_print_tran_strvec(nx_, &sx0, 0);
#endif

	// dynamic
	struct d_strmat sA; d_allocate_strmat(nx_, nx_, &sA);

	// 0 nominal + md realizations 1 to md
	struct d_strvec sb0[md+1];
	struct d_strvec sb1[md+1];
	struct d_strmat sBbt0[md+1];
	struct d_strmat sBAbt1[md+1];
	for(ii=0; ii<md+1; ii++)
		{
		d_allocate_strvec(nx_, &sb0[ii]);
		d_allocate_strvec(nx_, &sb1[ii]);
		d_allocate_strmat(nu_+1, nx_, &sBbt0[ii]);
		d_allocate_strmat(nu_+nx_+1, nx_, &sBAbt1[ii]);
		}

	// stage 0
	for(ii=0; ii<md+1; ii++)
		{
		d_cvt_mat2strmat(nx_, nx_, A+ii*nx_*nx_, nx_, &sA, 0, 0);
//		d_print_strmat(nx_, nx_, &sA, 0, 0);
		d_cvt_vec2strvec(nx_, b+ii*nx_, &sb1[ii], 0);
		dgemv_n_libstr(nx_, nx_, 1.0, &sA, 0, 0, &sx0, 0, 1.0, &sb1[ii], 0, &sb0[ii], 0);
//		d_print_tran_strvec(nx_, &sb0[ii], 0);
		d_cvt_tran_mat2strmat(nx_, nu_, B+ii*nx_*nu_, nx_, &sBbt0[ii], 0, 0);
		drowin_libstr(nx_, 1.0, &sb0[ii], 0, &sBbt0[ii], nu_, 0);
#if defined(PRINT)
		d_print_strmat(nu_+1, nx_, &sBbt0[ii], 0, 0);
#endif
		}

	// stage 1
	for(ii=0; ii<md+1; ii++)
		{
		d_cvt_vec2strvec(nx_, b+ii*nx_, &sb1[ii], 0);
		d_cvt_tran_mat2strmat(nx_, nu_, B+ii*nx_*nu_, nx_, &sBAbt1[ii], 0, 0);
		d_cvt_tran_mat2strmat(nx_, nx_, A+ii*nx_*nx_, nx_, &sBAbt1[ii], nu_, 0);
		d_cvt_tran_mat2strmat(nx_, 1, b+ii*nx_, nx_, &sBAbt1[ii], nu_+nx_, 0);
#if defined(PRINT)
		d_print_strmat(nu_+nx_+1, nx_, &sBAbt1[ii], 0, 0);
#endif
		}

	// cost function matrices
	struct d_strvec sr; d_allocate_strvec(nu_, &sr);
	d_cvt_vec2strvec(nu_, r, &sr, 0);

	struct d_strvec sq; d_allocate_strvec(nx_, &sq);
	d_cvt_vec2strvec(nx_, q, &sq, 0);

	struct d_strvec sp; d_allocate_strvec(nx_, &sp);
	d_cvt_vec2strvec(nx_, p, &sp, 0);

	struct d_strvec sdR; d_allocate_strvec(nu_, &sdR);
	d_cvt_vec2strvec(nu_, dR, &sdR, 0);

	struct d_strvec sdQ; d_allocate_strvec(nx_, &sdQ);
	d_cvt_vec2strvec(nx_, dQ, &sdQ, 0);

	struct d_strvec sdP; d_allocate_strvec(nx_, &sdP);
	d_cvt_vec2strvec(nx_, dP, &sdP, 0);

	// pack cost function
	struct d_strvec sr0; d_allocate_strvec(nu_, &sr0);
	dveccp_libstr(nu_, &sr, 0, &sr0, 0); // XXX no need to update r0 since S=0

	struct d_strmat sRr0; d_allocate_strmat(nu_+1, nu_, &sRr0);
	ddiain_libstr(nu_, 1.0, &sdR, 0, &sRr0, 0, 0);
	drowin_libstr(nu_, 1.0, &sr0, 0, &sRr0, nu_, 0);
#if defined(PRINT)
	d_print_strmat(nu_+1, nu_, &sRr0, 0, 0);
#endif

	struct d_strvec srq1; d_allocate_strvec(nu_+nx_, &srq1);
	dveccp_libstr(nu_, &sr, 0, &srq1, 0);
	dveccp_libstr(nx_, &sq, 0, &srq1, nu_);

	struct d_strmat sRSQrq1; d_allocate_strmat(nu_+nx_+1, nu_+nx_, &sRSQrq1);
	ddiain_libstr(nu_, 1.0, &sdR, 0, &sRSQrq1, 0, 0);
	ddiain_libstr(nx_, 1.0, &sdQ, 0, &sRSQrq1, nu_, nu_);
	drowin_libstr(nu_, 1.0, &sr, 0, &sRSQrq1, nu_+nx_, 0);
	drowin_libstr(nx_, 1.0, &sq, 0, &sRSQrq1, nu_+nx_, nu_);
#if defined(PRINT)
	d_print_strmat(nu_+nx_+1, nu_+nx_, &sRSQrq1, 0, 0);
#endif

	struct d_strvec sqN; d_allocate_strvec(nx_, &sqN);
	dveccp_libstr(nx_, &sp, 0, &sqN, 0);

	struct d_strmat sQqN; d_allocate_strmat(nx_+1, nx_, &sQqN);
	ddiain_libstr(nx_, 1.0, &sdP, 0, &sQqN, 0, 0);
	drowin_libstr(nx_, 1.0, &sp, 0, &sQqN, nx_, 0);
#if defined(PRINT)
	d_print_strmat(nx_+1, nx_, &sQqN, 0, 0);
#endif

	// bounds
	struct d_strvec sd0; d_allocate_strvec(2*nu_, &sd0);
	d_cvt_vec2strvec(nu_, umin, &sd0, 0);
	d_cvt_vec2strvec(nu_, umax, &sd0, nu_);
#if defined(PRINT)
	d_print_strvec(2*nu_, &sd0, 0);
#endif
	int *idxb0; int_zeros(&idxb0, nu_, 1);
	for(ii=0; ii<nu_; ii++)
		idxb0[ii] = ii;

	struct d_strvec sd1; d_allocate_strvec(2*nu_+2*nx_, &sd1);
	d_cvt_vec2strvec(nu_, umin, &sd1, 0);
	d_cvt_vec2strvec(nu_, umax, &sd1, nu_+nx_);
	d_cvt_vec2strvec(nx_, xmin, &sd1, nu_);
	d_cvt_vec2strvec(nx_, xmax, &sd1, nu_+nx_+nu_);
#if defined(PRINT)
	d_print_strvec(2*nu_+2*nx_, &sd1, 0);
#endif
	int *idxb1; int_zeros(&idxb1, nu_+nx_, 1);
	for(ii=0; ii<nu_+nx_; ii++)
		idxb1[ii] = ii;

	struct d_strvec sdN; d_allocate_strvec(2*nx_, &sdN);
	d_cvt_vec2strvec(nx_, xmin, &sdN, 0);
	d_cvt_vec2strvec(nx_, xmax, &sdN, nx_);
#if defined(PRINT)
	d_print_strvec(2*nx_, &sdN, 0);
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
	struct d_strmat hsBAbt[N];
	struct d_strvec hsb[N];
	struct d_strmat hsRSQrq[N+1];
	struct d_strvec hsrq[N+1];
	struct d_strvec hsdRSQ[N+1];
	struct d_strmat hsDCt[N+1];
	struct d_strvec hsd[N+1];
	struct d_strvec hsux[N+1];
	struct d_strvec hspi[N+1];
	struct d_strvec hslam[N+1];
	struct d_strvec hst[N+1];
	int *hidxb[N+1];
	// residuals
	struct d_strvec hsrrq[N+1];
	struct d_strvec hsrb[N];
	struct d_strvec hsrd[N+1];
	struct d_strvec hsrm[N+1];
	double mu;
	// work space
	struct d_strmat hsL[N+1];
	void *work_ipm;
	void *work_res;

	hsBAbt[0] = sBbt0[0];
	hsb[0] = sb0[0];
	hsRSQrq[0] = sRr0;
	hsrq[0] = sr0;
	hsd[0] = sd0;
	d_allocate_strvec(nu[0]+nx[0]+1, &hsux[0]);
	d_allocate_strvec(nx[0], &hspi[0]);
	d_allocate_strvec(2*nb[0]+2*ng[0], &hslam[0]);
	d_allocate_strvec(2*nb[0]+2*ng[0], &hst[0]);
	hidxb[0] = idxb0;
	d_allocate_strvec(nu[0]+nx[0], &hsrrq[0]);
	d_allocate_strvec(nx[1], &hsrb[0]);
	d_allocate_strvec(2*nb[0]+2*ng[0], &hsrd[0]);
	d_allocate_strvec(2*nb[0]+2*ng[0], &hsrm[0]);
	d_allocate_strmat(nu[0]+nx[0]+1, nu[0]+nx[0], &hsL[0]);
	for(ii=1; ii<N; ii++)
		{
		hsBAbt[ii] = sBAbt1[0];
		hsb[ii] = sb1[0];
		hsRSQrq[ii] = sRSQrq1;
		hsrq[ii] = srq1;
		hsd[ii] = sd1;
		d_allocate_strvec(nu[ii]+nx[ii]+1, &hsux[ii]);
		d_allocate_strvec(nx[ii], &hspi[ii]);
		d_allocate_strvec(2*nb[ii]+2*ng[ii], &hslam[ii]);
		d_allocate_strvec(2*nb[ii]+2*ng[ii], &hst[ii]);
		hidxb[ii] = idxb1;
		d_allocate_strvec(nu[ii]+nx[ii], &hsrrq[ii]);
		d_allocate_strvec(nx[ii+1], &hsrb[ii]);
		d_allocate_strvec(2*nb[ii]+2*ng[ii], &hsrd[ii]);
		d_allocate_strvec(2*nb[ii]+2*ng[ii], &hsrm[ii]);
		d_allocate_strmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], &hsL[ii]);
		}
	hsRSQrq[N] = sQqN;
	hsrq[N] = sqN;
	hsd[N] = sdN;
	d_allocate_strvec(nu[N]+nx[N]+1, &hsux[N]);
	d_allocate_strvec(nx[N], &hspi[N]);
	d_allocate_strvec(2*nb[N]+2*ng[N], &hslam[N]);
	d_allocate_strvec(2*nb[N]+2*ng[N], &hst[N]);
	hidxb[N] = idxbN;
	d_allocate_strvec(nu[N]+nx[N], &hsrrq[N]);
	d_allocate_strvec(2*nb[N]+2*ng[N], &hsrd[N]);
	d_allocate_strvec(2*nb[N]+2*ng[N], &hsrm[N]);
	d_allocate_strmat(nu[N]+nx[N]+1, nu[N]+nx[N], &hsL[N]);

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
	struct d_strmat t_hsBAbt[Nn-1]; // XXX defined on the edges !!!!!
	struct d_strvec t_hsb[Nn-1]; // XXX defined on the edges !!!!!
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
		d_print_strmat(t_nu[stage-1]+t_nx[stage-1]+1, t_nx[stage], &t_hsBAbt[ii], 0, 0);
		}
	return 0;
#endif

	// temporary cost function indexed by stage
	struct d_strmat tmp_hsRSQrq[Nh+1];
	struct d_strvec tmp_hsrq[Nh+1];
	// first stages: scale cost function
	for(ii=0; ii<Nr; ii++)
		{
		d_allocate_strmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], &tmp_hsRSQrq[ii]);
		d_allocate_strvec(nu[ii]+nx[ii], &tmp_hsrq[ii]);
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
		dgecp_libstr(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], &tmp_hsRSQrq[ii+1], 0, 0, &tmp_hsRSQrq[ii], 0, 0);
		dgesc_libstr(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], md, &tmp_hsRSQrq[ii], 0, 0);
		dveccp_libstr(nu[ii]+nx[ii], &tmp_hsrq[ii+1], 0, &tmp_hsrq[ii], 0);
		dvecsc_libstr(nu[ii]+nx[ii], md, &tmp_hsrq[ii], 0);
		}
	// scale first stage
	ii = 0;
	dgecp_libstr(nu[ii]+nx[ii], nu[ii]+nx[ii], &tmp_hsRSQrq[ii+1], 0, 0, &tmp_hsRSQrq[ii], 0, 0);
	dgesc_libstr(nu[ii]+nx[ii], nu[ii]+nx[ii], md, &tmp_hsRSQrq[ii], 0, 0);
	dgecp_libstr(1, nu[ii]+nx[ii], &tmp_hsRSQrq[ii+1], nu[ii+1]+nx[ii+1], 0, &tmp_hsRSQrq[ii], nu[ii]+nx[ii], 0);
	dgesc_libstr(1, nu[ii]+nx[ii], md, &tmp_hsRSQrq[ii], nu[ii]+nx[ii], 0);
	dveccp_libstr(nu[ii]+nx[ii], &tmp_hsrq[ii+1], 0, &tmp_hsrq[ii], 0);
	dvecsc_libstr(nu[ii]+nx[ii], md, &tmp_hsrq[ii], 0);

//	for(ii=0; ii<=Nh; ii++)
//		{
//		d_print_strmat(t_nu[ii]+t_nx[ii]+1, t_nu[ii]+t_nx[ii], &tmp_hsRSQrq[ii], 0, 0);
//		}
//	return;

	// cost function indexed by node
	// struct d_strmat t_hsRSQrq[Nn];
	// struct d_strvec t_hsrq[Nn];
	struct d_strmat *t_hsRSQrq = malloc(Nn*sizeof(struct d_strmat));
	struct d_strvec *t_hsrq = malloc(Nn*sizeof(struct d_strvec));
	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage;
		t_hsRSQrq[ii] = tmp_hsRSQrq[stage];
		t_hsrq[ii] = tmp_hsrq[stage];
		}

	// constraints indexed by node
	// struct d_strmat t_hsDCt[Nn];
	// struct d_strvec t_hsd[Nn];
	int *t_hidxb[Nn];

	struct d_strmat *t_hsDCt = malloc(Nn*sizeof(struct d_strmat));
	struct d_strvec *t_hsd = malloc(Nn*sizeof(struct d_strvec));

	for(ii=0; ii<Nn; ii++)
		{
		stage = tree[ii].stage;
		t_hsd[ii] = hsd[stage];
		t_hidxb[ii] = hidxb[stage];
		}

	// res
	struct d_strvec *t_hsrrq = malloc(Nn*sizeof(struct d_strvec));
	struct d_strvec *t_hsrb = malloc((Nn-1)*sizeof(struct d_strvec));
	struct d_strvec *t_hsrd = malloc(Nn*sizeof(struct d_strvec));
	struct d_strvec *t_hsrm = malloc(Nn*sizeof(struct d_strvec));

	// work
	struct d_strmat *t_hsL = malloc(Nn*sizeof(struct d_strmat));
	void *t_work_res;
	for(ii=0; ii<Nn; ii++)
		{
		d_allocate_strvec(t_nu[ii]+t_nx[ii], &t_hsrrq[ii]);
		d_allocate_strvec(2*t_nb[ii]+2*t_ng[ii], &t_hsrd[ii]);
		d_allocate_strvec(2*t_nb[ii]+2*t_ng[ii], &t_hsrm[ii]);
		d_allocate_strmat(t_nu[ii]+t_nx[ii]+1, t_nu[ii]+t_nx[ii], &t_hsL[ii]);
		for(jj=0; jj<tree[ii].nkids; jj++)
			{
			idxkid = tree[ii].kids[jj];
			d_allocate_strvec(t_nx[idxkid], &t_hsrb[idxkid-1]);
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

int_t idxp, idxb;
for (int_t kk = 0; kk < qp_in.N; kk++) {

	// TODO(dimitris): Add S' (nx x nu) term to lower diagonal part
	dgecp_libstr(qp_in.nu[kk], qp_in.nu[kk], (struct d_strmat *)&qp_in.R[kk], 0, 0, &work.sRSQrq[kk], 0, 0);
	dgecp_libstr(qp_in.nx[kk], qp_in.nx[kk], (struct d_strmat *)&qp_in.Q[kk], 0, 0, &work.sRSQrq[kk], qp_in.nu[kk], qp_in.nu[kk]);

	drowin_libstr(qp_in.nu[kk], 1.0, (struct d_strvec *)&qp_in.r[kk], 0, &work.sRSQrq[kk], qp_in.nu[kk] + qp_in.nx[kk], 0);
	drowin_libstr(qp_in.nx[kk], 1.0, (struct d_strvec *)&qp_in.q[kk], 0, &work.sRSQrq[kk], qp_in.nu[kk] + qp_in.nx[kk], qp_in.nu[kk]);

	if (kk > 0) {
		idxp = tree[kk].dad;
		dgetr_libstr(qp_in.nx[kk], qp_in.nu[idxp], (struct d_strmat *)&qp_in.B[kk-1], 0, 0, &work.sBAbt[kk-1], 0, 0);
		dgetr_libstr(qp_in.nx[kk], qp_in.nx[idxp], (struct d_strmat *)&qp_in.A[kk-1], 0, 0, &work.sBAbt[kk-1], qp_in.nu[idxp], 0);
		drowin_libstr(qp_in.nx[kk], 1.0, (struct d_strvec *)&qp_in.b[kk-1], 0, &work.sBAbt[kk-1], qp_in.nx[idxp] + qp_in.nu[idxp], 0);
	}

	for (int_t jj = 0; jj < work.nb[kk]; jj++) {
		idxb = work.idxb[kk][jj];
		if (idxb < qp_in.nu[kk]) {
			DVECEL_LIBSTR(&work.sd[kk], jj) = DVECEL_LIBSTR(&qp_in.umin[kk], idxb);
			DVECEL_LIBSTR(&work.sd[kk], jj + work.nb[kk]) = DVECEL_LIBSTR(&qp_in.umax[kk], idxb);
		} else {
			DVECEL_LIBSTR(&work.sd[kk], jj) = DVECEL_LIBSTR(&qp_in.xmin[kk], idxb - qp_in.nu[kk]);
			DVECEL_LIBSTR(&work.sd[kk], jj + work.nb[kk]) = DVECEL_LIBSTR(&qp_in.xmax[kk], idxb - qp_in.nu[kk]);
		}
	}
}

// *****************************************************************************

#if 0
for(ii=0; ii<N; ii++)
	d_print_strmat(nu[ii]+nx[ii]+1, nx[ii+1], &t_hsBAbt[ii], 0, 0);
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
	dveccp_libstr(qp_in.nu[ii], &work.sux[ii], 0, &qp_out.u[ii], 0);
	dveccp_libstr(qp_in.nx[ii], &work.sux[ii], qp_in.nu[ii], &qp_out.x[ii], 0);
}

// *****************************************************************************


#if defined(PRINT)
	printf("\n... done\n\n");

	// print sol
	printf("\nt_ux = \n\n");
	for(ii=0; ii<Nn; ii++)
		{
		d_print_tran_strvec(t_nu[ii]+t_nx[ii], &work.sux[ii], 0);
		}
	printf("\nt_pi = \n\n");
	for(ii=0; ii<Nn; ii++)
		{
		d_print_tran_strvec(t_nx[ii], &qp_out.lam[ii], 0);
		}

	// // print res
	// printf("\nt_res_rq\n");
	// for(ii=0; ii<Nn; ii++)
	// 	d_print_e_tran_strvec(t_nu[ii]+t_nx[ii], &t_hsrrq[ii], 0);

	// printf("\nt_res_b\n");
	// for(ii=0; ii<Nn-1; ii++)
	// 	d_print_e_tran_strvec(t_nx[ii+1], &t_hsrb[ii], 0);

	// printf("\nt_res_d\n");
	// for(ii=0; ii<Nn; ii++)
	// 	d_print_e_tran_strvec(2*t_nb[ii]+2*t_ng[ii], &t_hsrd[ii], 0);

	// printf("\nt_res_m\n");
	// for(ii=0; ii<Nn; ii++)
	// 	d_print_e_tran_strvec(2*t_nb[ii]+2*t_ng[ii], &t_hsrm[ii], 0);

	printf("\nstatistics from last run\n\n");
	for(jj=0; jj<qp_out.info.iter; jj++)
		printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\talpha = %f\tmu = %f\tmu = %e\n", jj, work.status[5*jj], work.status[5*jj+1], work.status[5*jj+2], work.status[5*jj+2], work.status[5*jj+3], work.status[5*jj+4], work.status[5*jj+4]);
	printf("\n");

	printf("\ntime tree ipm\n");
	printf("\n%e\n", time_tree_ipm);
	printf("\n");

	// print last element of sol
	printf("\nt_ux[Nn] = \n\n");
	d_print_tran_strvec(t_nu[Nn-1]+t_nx[Nn-1], &work.sux[Nn-1], 0);
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
		fprintf(xfile, "%1.15f\n", dvecex1_libstr(&sx0, jj));
	for(ii=0; ii<Nn; ii++) // loop over nodes
		{
		for(jj=0; jj<t_nx[ii]; jj++)
			fprintf(xfile, "%1.15f\n", dvecex1_libstr(&work.sux[ii], t_nu[ii]+jj));
		for(jj=0; jj<t_nu[ii]; jj++)
			fprintf(xfile, "%1.15f\n", dvecex1_libstr(&work.sux[ii], jj));
		}

	fclose(tfile);
	fclose(xfile);
	fclose(ifile);

/************************************************
*	free memory
************************************************/

	// IPM common

	// nominal MPC
	d_free_strvec(&sx0);
	d_free_strmat(&sA);
	d_free_strvec(&sr);
	d_free_strvec(&sq);
	d_free_strvec(&sp);
	d_free_strvec(&sdR);
	d_free_strvec(&sdQ);
	d_free_strvec(&sdP);
	d_free_strvec(&sr0);
	for(ii=0; ii<md+1; ii++)
		{
		d_free_strvec(&sb0[ii]);
		d_free_strvec(&sb1[ii]);
		d_free_strmat(&sBbt0[ii]);
		d_free_strmat(&sBAbt1[ii]);
		}
	for(ii=0; ii<Nr; ii++)
		{
		d_free_strmat(&tmp_hsRSQrq[ii]);
		d_free_strvec(&tmp_hsrq[ii]);
		}
	d_free_strmat(&sRr0);
	d_free_strvec(&srq1);
	d_free_strmat(&sRSQrq1);
	d_free_strvec(&sqN);
	d_free_strmat(&sQqN);
	d_free_strvec(&sd0);
	d_free_strvec(&sd1);
	d_free_strvec(&sdN);
	int_free(idxb0);
	int_free(idxb1);
	int_free(idxbN);
	d_free_align(work_ipm);
	d_free_align(work_res);
	for(ii=0; ii<N; ii++)
		{
		d_free_strvec(&hsux[ii]);
		d_free_strvec(&hspi[ii]);
		d_free_strvec(&hslam[ii]);
		d_free_strvec(&hst[ii]);
		d_free_strvec(&hsrrq[ii]);
		d_free_strvec(&hsrb[ii]);
		d_free_strvec(&hsrd[ii]);
		d_free_strvec(&hsrm[ii]);
		d_free_strmat(&hsL[ii]);
		}
	ii = N;
	d_free_strvec(&hsux[ii]);
	d_free_strvec(&hspi[ii]);
	d_free_strvec(&hslam[ii]);
	d_free_strvec(&hst[ii]);
	d_free_strvec(&hsrrq[ii]);
	d_free_strvec(&hsrd[ii]);
	d_free_strvec(&hsrm[ii]);
	d_free_strmat(&hsL[ii]);

	// tree MPC with tree HPMPC
	d_free_align(t_work_res);
	for(ii=0; ii<Nn; ii++)
		{
		d_free_strmat(&t_hsL[ii]);
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
