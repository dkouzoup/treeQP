function logging = treeqp_main(benchmark, sim_opts, solver_opts)

%% Initialize

clc; close all;
clearvars -except benchmark sim_opts solver_opts;
addpath('utils', 'common');

if verLessThan('matlab', 'R2016a')
    error('Matlab version >= R2016a required');
end

%% Choose default options

if nargin == 0
    rng('default') % set seed

    % default benchmark problem
    benchmark = 'crane';

    % choose simulation options

    sim_opts.Nh      = 50;   % prediction horizon
    sim_opts.Nr      = 2;    % robust horizon (<= Nh)
    sim_opts.md      = 3;    % number of realizations
    sim_opts.nMPC    = 50;   % number of closed-loop MPC steps
    sim_opts.NRUNS   = 11;   % run NRUNS times and take minimum cputime per MPC problem
    sim_opts.KKT_TOL = 1e-6; % pause simulation if solvers returns inaccurate solution

    % choose and customize solver

    solver_opts.name = 'tdunes'; % choose algoritm (tdunes, sdunes, hpmpc)

    solver_opts.BLASFEO_LA     = 'HIGH_PERFORMANCE';
    solver_opts.BLASFEO_TARGET = 'X64_INTEL_SANDY_BRIDGE';

    % solver-specific options

    switch solver_opts.name

        case 'tdunes'

            solver_opts.maxIter            = 100;  % maximum number of dual Newton iterations
            solver_opts.termTolerance      = 1e-6; % tolerance for termination condition (inf norm of residuals)

            solver_opts.linesearch.alg     = 'Armijo_with_backtracking';
            solver_opts.linesearch.maxIter = 100;  % maximum LS iterations
            solver_opts.linesearch.gamma   = 0.1;  % relaxation of gradient descent
            solver_opts.linesearch.beta    = 0.6;  % tau reduction

            solver_opts.openmp.ON          = 0;    % use openmp
            solver_opts.openmp.nthreads    = 2;    % number of threads if openmp is ON

            solver_opts.reg.mode           = 'ON_THE_FLY_LEVENBERG_MARQUARDT';  % 'NO_REGULARIZATION' or 'ALWAYS_LEVENBERG_MARQUARDT' or 'ON_THE_FLY_LEVENBERG_MARQUARDT'
            solver_opts.reg.value          = 1e-8;
            solver_opts.reg.tol            = 1e-8;

            solver_opts.WARMSTART             = 0;
            solver_opts.CHECK_LAST_ACTIVE_SET = 1;
            solver_opts.DETAILED_TIMINGS      = 0;

        case 'sdunes'

            solver_opts.maxIter            = 100;
            solver_opts.termTolerance      = 1e-6;

            solver_opts.linesearch.alg     = 'Armijo_with_backtracking';
            solver_opts.linesearch.maxIter = 100;
            solver_opts.linesearch.gamma   = 0.1;
            solver_opts.linesearch.beta    = 0.6;

            solver_opts.openmp.ON          = 0;
            solver_opts.openmp.nthreads    = 2;

            solver_opts.reg.mode           = 'ON_THE_FLY_LEVENBERG_MARQUARDT';
            solver_opts.reg.value          = 1e-8;
            solver_opts.reg.tol            = 1e-8;

            solver_opts.WARMSTART = 1;
            solver_opts.CHECK_LAST_ACTIVE_SET = 1;
            solver_opts.DETAILED_TIMINGS = 0;

        case 'hpmpc'

            solver_opts.maxIter   = 50;     % maximum number of IP iterations
            solver_opts.alpha_min = 1e-8;
            solver_opts.mu_tol    = 1e-8;

    end

    matlab_opts.SOLVE_QUADPROG = 0;  % solve with quadprog to compare results

else
    matlab_opts.SOLVE_QUADPROG = 0;
end

if ~benchmark_exists(benchmark)
    error('Specified benchmark problem does not exist')
end

nMPC = sim_opts.nMPC;
Nh   = sim_opts.Nh;
Nr   = sim_opts.Nr;
md   = sim_opts.md;

% import benchmark
[dynamics, weights, constraints, nominal] = import_benchmark(benchmark, md, sim_opts.nMPC);

% store problem dimensions
[nx, nu] = size(dynamics(1).B);

% Set quadprog options
if matlab_opts.SOLVE_QUADPROG
    quadprog_opts = optimoptions('quadprog','Display','off');
    quadprog_opts.TolCon = 1e-16;
    quadprog_opts.TolFun = 1e-16;
    quadprog_opts.TolX   = 1e-16;
end

% build tree
agents = setup_tree(dynamics, weights, constraints, Nr, Nh, md);

%% Compile treeQP

TREEQP_ROOT = [pwd '/../'];

treeqp_compile(solver_opts, sim_opts.NRUNS, TREEQP_ROOT);

% TODO: check if still works
if md == 1
    save_dimensions_nominal_qpdunes(Nh, solver_opts.maxIter, dynamics, weights, constraints, []);
end

%% Closed-loop simulation

% initialize logging
logging.cpuTime = nan(1, nMPC);
logging.iter    = nan(1, nMPC);
logging.status  = nan(1, nMPC);
logging.kkt     = nan(1, nMPC);

trajectories{nMPC} = [];
xMPC  = nan(nMPC+1,nx);
uMPC  = nan(nMPC,nu);

for iMPC = 1:nMPC

    % update time-varying reference (for quadcopter example)
    if isfield(weights, 'xref')
        weights.q = -weights.Q.*weights.xref(:, iMPC);
        weights.r = -weights.R.*weights.uref(:, iMPC);
        weights.p = weights.q;
        % code snippet from setup_tree
        for ii = 1:length(agents)
            if ~isempty(agents(ii).child)
                if agents(ii).stage <= Nr
                    w = md^(Nr-agents(ii).stage);
                else
                    w = 1;
                end
                agents(ii).h = w*[weights.q; weights.r];
            else
                agents(ii).h = weights.p;
            end
        end
    end

    % store closed-loop state trajectory
    xMPC(iMPC,:) = constraints.x0';

    % Solve tree QP
    [agents, info] = treeqp_solve(agents, TREEQP_ROOT, solver_opts, iMPC);

    if info.status ~= 0
        disp(['treeQP failed with status ' num2str(info.status) ' (KKT ' num2str(info.kkt) ')']);
        keyboard;
    end
    if info.kkt > sim_opts.KKT_TOL
        disp(['treeQP failed to meet required accuracy (KKT ' num2str(info.kkt) ')']);
        % keyboard;
    end

    % build trajectories based on optimal primal iterates
    trajectories{iMPC} = build_trajectories_from_trees(agents, Nh, Nr);

    % store closed-loop control trajectory extract optimized control
    uMPC(iMPC,:) = agents(1).uit(1:nu)';

    % check with optimal solution returned by quadprog
    if matlab_opts.SOLVE_QUADPROG
        trajectories_quadprog = solve_trees_quadprog(agents, Nh, Nr, quadprog_opts);
        error_treeqp_quadprog = max(check_error_in_trajectories(trajectories_quadprog, trajectories{iMPC}));
        % disp(['ERROR FROM QUADPROG = ' num2str(error_treeqp_quadprog) '.']);
        if error_treeqp_quadprog > 1e-3
            disp(uMPC(iMPC,:)');
            disp(['ERROR LARGER THAN THRESHOLD (' num2str(error_treeqp_quadprog) ' - KKT ' num2str(info.kkt) ') . CONTINUE?']);
            keyboard
        end
    end

    % print info
    if iMPC == 1
        fprintf('\n****************************************************************\n  ');
        fprintf('                          C code stats                                ')
        fprintf('\n****************************************************************\n\n');
    end
    if matlab_opts.SOLVE_QUADPROG == 1
        if  mod(iMPC, 10) == 0 || iMPC == 1
            fprintf(' MPC ||  ITER  ||    TIME   ||      KKT     || QUADPROG \n');
        end
        fprintf('%3d  ||  %3d   ||  %5.2f ms ||  %2.2e    || %2.2e \n', iMPC, info.iter, 1000*info.cpuTime, info.kkt, error_treeqp_quadprog);
    else
        if  mod(iMPC, 10) == 0 || iMPC == 1
            fprintf(' MPC ||  ITER  ||    TIME   ||      KKT\n');
        end
        fprintf('%3d  ||  %3d   ||  %5.2f ms ||  %2.2e\n', iMPC, info.iter, 1000*info.cpuTime, info.kkt);
    end
    if iMPC == nMPC
        fprintf('\n****************************************************************\n\n\n');
    end

    % apply optimized control to simulator
    if strcmp(benchmark, 'quadcopter')
        if ~exist('qMPC', 'var')
            qMPC = nan(nMPC, 1); % initialization of quartenion
            qMPC(1) = 1;
        end

        x0_aug = nominal.simulate([qMPC(iMPC); xMPC(iMPC,:)'], uMPC(iMPC,:)');

        qMPC(iMPC+1) = x0_aug(1);

        constraints.x0 = x0_aug(2:end);
    elseif isfield(nominal, 'simulate')
        constraints.x0 = nominal.simulate(xMPC(iMPC,:)', uMPC(iMPC,:)');
    else
        constraints.x0 = nominal.A*xMPC(iMPC,:)' + nominal.B*uMPC(iMPC,:)' + nominal.c;
    end

    % embed new initial value
    agents(1).zmin(1:nx) = constraints.x0;
    agents(1).zmax(1:nx) = constraints.x0;

    % log results
    logging.cpuTime(iMPC) = info.cpuTime;
    logging.iter(iMPC)    = info.iter;
    logging.status(iMPC)  = info.status;
    logging.kkt(iMPC)     = info.kkt;

    if isfield(info, 'timings')
        logging.timings(iMPC) = info.timings;
    end

end

% Store terminal state
xMPC(end,:) = constraints.x0;

% finish logging
logging.trajectories = trajectories;
logging.solver_opts  = solver_opts;

logging.xMPC = xMPC;
logging.uMPC = uMPC;

logging.Nh = Nh;
logging.Nr = Nr;
logging.md = md;
logging.nx = nx;
logging.nu = nu;

% TODO: move somewhere else
if strcmp(benchmark, 'quadcopter') && nargin == 0
    Ts = 0.05;
    eul_cl = zeros(3, nMPC);

    xMPCaug = [qMPC xMPC];
    for i = 1:nMPC
        eul_cl(:, i) = myquat2eul(xMPCaug(i, 1:4)').';
    end
    figure()
    subplot(311)
    hold all
    plot([0:nMPC-1]*Ts, eul_cl(1,:))
    plot([0:nMPC-1]*Ts, weights.eulref(1,1:nMPC), '--')
    grid on
    ylabel('roll')
    xlabel('times [s]')
    legend('mpc', 'ref')

    subplot(312)
    hold all
    plot([0:nMPC-1]*Ts, eul_cl(2,:))
    plot([0:nMPC-1]*Ts, weights.eulref(2,1:nMPC), '--')
    grid on
    ylabel('pitch')
    xlabel('times [s]')
    legend('mpc','ref')

    subplot(313)
    hold all
    plot([0:nMPC-1]*Ts, eul_cl(3,:))
    plot([0:nMPC-1]*Ts, weights.eulref(3,1:nMPC), '--')
    grid on
    xlabel('times [s]')
    ylabel('yaw')
    legend('mpc','ref')
    ylim([-1;1])
end

if nargin == 0
    % stop here to check results if not called from another script
    figure
    subplot(2,1,1); plot(xMPC);
    subplot(2,1,2); plot(uMPC);
    keyboard
end

end
