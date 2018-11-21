function [agents, info] = treeqp_solve(agents, TREEQP_ROOT, solver_opts, iMPC, GENERATE_QP_IN)

% TREEQP_SOLVE write QP (or only x0) on json file and call treeQP to solve
%              the problem. The solution is also returned as a json file.

persistent LAM MU; % to store multipliers between calls in dual Newton algorithms
persistent HPREV;  % to store previous gradient and recompile if changed

%% check whether to re-generate qp_in.json

if nargin < 5
    if iMPC == 1 || max(abs(agents(1).h - HPREV)) > 0
        GENERATE_QP_IN = true;
    else
        GENERATE_QP_IN = false;
    end
end

%% initialize input/output

FNAME_IN = 'qp_in.json';

if GENERATE_QP_IN
    addpath([TREEQP_ROOT '/examples/random_qp_utils']); % for code_generate_json function

    if exist(FNAME_IN, 'file')
        delete(FNAME_IN)
    end
end

FNAME_OUT = 'qp_out.json';

if exist(FNAME_OUT, 'file')
    delete(FNAME_OUT)
end

%% create agents struct to code generate json

if GENERATE_QP_IN

    agents_mod = agents;

    for ii = 1:length(agents)

        if ~isempty(agents(ii).child)
            kid = agents(ii).child(1);
            nx = size(agents(kid).dyn.A, 2);
        else
            nx = size(agents(ii).dyn.A, 1);
        end

        agents_mod(ii).Q = agents_mod(ii).H(1:nx, 1:nx);
        agents_mod(ii).R = agents_mod(ii).H(nx+1:end, nx+1:end);
        agents_mod(ii).S = agents_mod(ii).H(1:nx, nx+1:end)';

        agents_mod(ii).q = agents_mod(ii).h(1:nx);
        agents_mod(ii).r = agents_mod(ii).h(nx+1:end);

        agents_mod(ii).xmin = agents_mod(ii).zmin(1:nx);
        agents_mod(ii).xmax = agents_mod(ii).zmax(1:nx);
        agents_mod(ii).umin = agents_mod(ii).zmin(nx+1:end);
        agents_mod(ii).umax = agents_mod(ii).zmax(nx+1:end);

        if ii > 1
            agents_mod(ii).A = agents_mod(ii).dyn.A;
            agents_mod(ii).B = agents_mod(ii).dyn.B;
            agents_mod(ii).b = agents_mod(ii).dyn.c;
        end

        agents_mod(ii).dad = agents(ii).parent;
    end

end

%% setup options

if GENERATE_QP_IN

    % common options
    % TODO: use same opts directly in main script instead of this cast
    treeqp_opts.maxit = solver_opts.maxIter;

    % dual Newton options
    if contains(solver_opts.name, 'dunes')

        treeqp_opts.warmstart = solver_opts.WARMSTART;

        treeqp_opts.stationarityTolerance = solver_opts.termTolerance;

        treeqp_opts.lineSearchMaxIter = solver_opts.linesearch.maxIter;
        treeqp_opts.lineSearchBeta    = solver_opts.linesearch.beta;
        treeqp_opts.lineSearchGamma   = solver_opts.linesearch.gamma;

        treeqp_opts.regType  = ['TREEQP_' solver_opts.reg.mode];
        treeqp_opts.regTol   = solver_opts.reg.tol;
        treeqp_opts.regValue = solver_opts.reg.value;

        treeqp_opts.checkLastActiveSet = solver_opts.CHECK_LAST_ACTIVE_SET;

    end

    % solver-specific options
    switch solver_opts.name

        case 'tdunes'

            treeqp_opts.solver    = 'tdunes';
            treeqp_opts.clipping  = true;

        case 'sdunes'

            treeqp_opts.solver    = 'sdunes';

        case 'hpmpc'

            treeqp_opts.solver    = 'hpmpc';
            treeqp_opts.mu_tol    = solver_opts.mu_tol;
            treeqp_opts.alpha_min = solver_opts.alpha_min;
    end

end

%% code generate x0 and initialization of multipliers

nx0  = size(agents(2).dyn.A, 2);
tmps = struct('x0', agents(1).zmin(1:nx0));

% write multipliers in json (initialized to zero at first iteration)

switch solver_opts.name

    case 'tdunes'

        if iMPC == 1
            nx      = size(agents(2).dyn.B, 1);
            Nn      = length(agents);
            dim_lam = (Nn-1)*nx;
            LAM     = zeros(dim_lam, 1);
        end
        tmps.lam0_tree = LAM;

    case 'sdunes'

        if iMPC == 1

            % calculate multi-stage MPC parameters
            md = length(agents(1).child);
            Nh = agents(end).stage;
            nx = size(agents(2).dyn.B, 1);
            nu = size(agents(2).dyn.B, 2);
            for ii = 1:length(agents)
                if length(agents(ii).child) == 1
                    Nr = agents(ii).stage;
                    break;
                end
            end
            Ns = md^Nr;

            if (Ns == 1)
                dim_lam = 0;
            else
                dim_lam = (Nr*Ns - (Ns-1)/(md-1))*nu;
            end

            dim_mu = Ns*Nh*nx;

            % initialize multipliers to zero for first iteration
            LAM = zeros(dim_lam, 1);
            MU  = zeros(dim_mu, 1);
        end
        tmps.lam0_scen = LAM;
        tmps.mu0_scen  = MU;
end

% encode date to json file
tmpj = jsonencode(tmps);
tmpn = 'qp_init.json';
tmpf = fopen(tmpn, 'w');
fprintf(tmpf, tmpj);
fclose(tmpf);

%% code generate input run executable

if GENERATE_QP_IN
    code_generate_json(FNAME_IN, agents_mod, treeqp_opts);
end

system(['./solve_qp_json.out ' FNAME_IN ' ' tmpn ' > ' FNAME_OUT]);

%% read output and save info

fid = fopen(FNAME_OUT);
con = fscanf(fid, '%s');
res = jsondecode(con);

info.iter    = res.info.num_iter;
info.cpuTime = res.info.cpu_time;
info.kkt     = res.info.kkt_tol;
info.status  = res.info.status;

if res.info.solver ~= solver_opts.name
    error('wrong solver used in treeQP')
end

if contains(solver_opts.name, 'dunes')
    if isfield(res.info, 'ls_iters')
        info.ls_iters = res.info.ls_iters;
    else
        info.ls_iters = nan; % need to set PROFILE = 2 in treeqp_compile
    end
end


if contains(solver_opts.name, 'dunes') && solver_opts.DETAILED_TIMINGS
    info.timings.solveStageQPs = res.info.cpu_times_stage_qps;
    info.timings.setupNewtonSystem = res.info.cpu_times_dual_system;
    info.timings.findNewtonDirection = res.info.cpu_times_newton_direction;
    info.timings.lineSearch = res.info.cpu_times_line_search;
    info.cpuTime = nan;
end

fclose(fid);

%% write solution to agents

% TODO: extract multipliers of ineq. constraints

for ii = 1:length(agents)

    agents(ii).xit = res.solution.nodes(ii).x;

    if ~isempty(agents(ii).child)
        agents(ii).uit = res.solution.nodes(ii).u;
    end

    if ii > 1
        agents(ii).lambda = res.solution.edges(ii-1).lam;
    end
end

%% prepare multipliers for next iteration

switch solver_opts.name

    case 'tdunes'

        LAM = res.init.lam0_tree;

        LAM = update_multipliers_tree(LAM, agents, solver_opts.WARMSTART);

    case 'sdunes'

        MU  = res.init.mu0_scen;
        LAM = res.init.lam0_scen;

        [LAM, MU] = update_multipliers_scenarios(LAM, MU, solver_opts.WARMSTART);

end

HPREV = agents(1).h;

end


function lam0 = update_multipliers_tree(lam0, agents, warmstart_mode)

if warmstart_mode == 0

    lam0 = 0*lam0;

elseif warmstart_mode == 1

    % keep current values for multipliers

elseif warmstart_mode == 2

    % perform shifting
    error('shifting for tree is broken')

    %     ------------ OLD CODE ------------
    %
    %             % find on which child of the root is the new state of the system closer
    %             min_dist = inf;
    %             min_indx = -1;
    %             for kids = agents(1).child
    %                 dist = norm(agents(kids).xit - constraints.x0, 2);
    %                 if dist < min_dist
    %                     min_dist = dist;
    %                     min_indx = kids;
    %                 end
    %             end
    %
    %             % set up indices to update lambdas
    %             advanced_ptrs = agents(min_indx).child;
    %             regular_ptrs  = agents(1).child;
    %             if length(advanced_ptrs) == 1
    %                 advanced_ptrs = repmat(advanced_ptrs, 1, length(regular_ptrs));
    %             end
    %
    %             % update lamda through tree
    %             while 1
    %
    %                 if isempty(advanced_ptrs)
    %
    %                     % NOTE: using 0 or parent's lambda doesn't really improve anything
    %                     for ii = 1:length(regular_ptrs)
    %                         % agents(regular_ptrs(ii)).lambda = 0*agents(regular_ptrs(ii)).lambda;
    %                         % agents(regular_ptrs(ii)).lambda = agents(agents(regular_ptrs(ii)).parent).lambda;
    %                     end
    %
    %                     break;
    %                 end
    %
    %                 new_regular_ptrs  = [];
    %                 new_advanced_ptrs = [];
    %
    %                 for ii = 1:length(regular_ptrs)
    %                     % assign lambda
    %                     %disp(['assigning lambda_' num2str(advanced_ptrs(ii)) ' to lambda_' num2str(regular_ptrs(ii))]);
    %                     agents(regular_ptrs(ii)).lambda = agents(advanced_ptrs(ii)).lambda;
    %                     % accumulate children
    %                     regular_children  = agents(regular_ptrs(ii)).child;
    %                     new_regular_ptrs  = [new_regular_ptrs regular_children];
    %                     advanced_children = agents(advanced_ptrs(ii)).child;
    %                     if length(advanced_children) == 1
    %                         advanced_children = repmat(advanced_children, 1, length(regular_children));
    %                     end
    %                     new_advanced_ptrs = [new_advanced_ptrs advanced_children];
    %                 end
    %
    %                 regular_ptrs  = new_regular_ptrs;
    %                 advanced_ptrs = new_advanced_ptrs;
    %
    %             end

else
    error('Unknown warm-start flag');
end


end



function [lam0, mu0] = update_multipliers_scenarios(lam0, mu0, warmstart_mode)

if warmstart_mode == 0

    % set everything to zero
    mu0  = 0*mu0;
    lam0 = 0*lam0;

elseif warmstart_mode == 1

    % keep the same multipliers

elseif warmstart_mode == 2 || warmstart_mode == 3 || warmstart_mode == 4

    % perform shifting
    error('shifting for scenarios is broken')

    %     ------------ OLD CODE ------------
    %
    %     % define how many scenarios on each branch of the root
    %     Nbranch = md^(Nr-1);
    %
    %     % find the root branch that is closer to current x0
    %     xits  = [scenarios.xit];
    %     x1s   = xits(1:nx,1:Nbranch:end);
    %     xdist = x1s - repmat(constraints.x0,1,md);
    %     xnorm = diag(sqrt(xdist'*xdist))';
    %     [~, xnearest] = min(xnorm);
    %
    %     mus = [scenarios.mu];  % gather multipliers
    %     mus = mus(nx+1:end,:); % trhow away first stage
    %     mus = mus(:,(xnearest-1)*Nbranch+1:xnearest*Nbranch); % keep only closest branch
    %     mus = [mus; mus(end-nx+1:end,:)]; % repeat multipliers of last state
    %
    %     if solver_opts.WARMSTART == 2
    %         mus = repmat(mus,1,md);
    %     elseif solver_opts.WARMSTART == 3
    %         mus = repmat(mus,md,1);
    %         mus = reshape(mus,nx*Nh,Nscenarios);
    %     elseif solver_opts.WARMSTART == 4
    %         mus = repmat(mus,md,1);
    %         mus = reshape(mus,nx*Nh,Nscenarios);
    %         mus(1:nx,:) = mus(1:nx,:)./md;
    %     end
    %
    %     mus = mat2cell(mus,size(mus,1),ones(Nscenarios,1)); % convert to cell
    %
    %     [scenarios.mu] = mus{:};

else
    error('Unknown warm-start flag');
end

end