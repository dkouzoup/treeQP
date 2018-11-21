
clear all; close all; clc

% create filename for workspace
t = clock;
t = t(1:end-1);
t = mat2str(t);
t = t(2:end-1);
t(t == ' ') = '_';
t = ['PERFORMANCE_PLOT_' t];

% open status file
STOREPATH = './';
fid       = fopen([STOREPATH 'status.txt'],'w');

RUN_TREE      = 1;
RUN_SCENARIOS = 1;
RUN_HPMPC     = 1;
EXPORT_FIG    = 0;
DROP_AS_PLOT  = 1;

%% Benchmark models

models = {...
    'linear_chain_nm_2_nu_1', ...
    'linear_chain_nm_4_nu_3', ...
    'linear_chain_nm_8_nu_7', ...
    'quadcopter', ...
    'crane'
    };

%% Design experiments

range_Nh = 20:10:50;
range_Nr = 1:4;
range_md = 2:4;

NH = repmat(range_Nh, length(range_Nr)*length(range_md), 1);
NH = NH(:)';
NR = repmat(range_Nr, length(range_md), 1);
NR = repmat(NR(:)', 1, length(range_Nh));
MD = repmat(range_md, 1, length(range_Nh)*length(range_Nr));

%% Simulation options

sim_opts.Nh      = NaN;
sim_opts.Nr      = NaN;
sim_opts.md      = NaN;
sim_opts.nMPC    = 50;
sim_opts.NRUNS   = 1;%10;
sim_opts.KKT_TOL = 1e-6;

%% tree DUNES options

tree_opts.name                  = 'tdunes';

tree_opts.maxIter               = 200;
tree_opts.termTolerance         = 1e-8;
tree_opts.DETAILED_TIMINGS      = 0;

tree_opts.openmp.ON             = 0;
tree_opts.openmp.nthreads       = NaN;

tree_opts.reg.mode              = 'ALWAYS_LEVENBERG_MARQUARDT';
tree_opts.reg.tol               = 1e-8;
tree_opts.reg.value             = 1e-8;

tree_opts.linesearch.alg        = 'Armijo_with_backtracking';
tree_opts.linesearch.maxIter    = 60;
tree_opts.linesearch.gamma      = 0.1;
tree_opts.linesearch.beta       = 0.6;

tree_opts.BLASFEO_LA            = 'REFERENCE';
tree_opts.BLASFEO_TARGET        = 'X64_INTEL_SANDY_BRIDGE';

tree_opts.CHECK_LAST_ACTIVE_SET = 1;

%% scenario DUNES options

scen_opts.name                  = 'sdunes';

scen_opts.maxIter               = tree_opts.maxIter;
scen_opts.termTolerance         = tree_opts.termTolerance;
scen_opts.DETAILED_TIMINGS      = tree_opts.DETAILED_TIMINGS;

scen_opts.BLASFEO_LA            = tree_opts.BLASFEO_LA;
scen_opts.BLASFEO_TARGET        = tree_opts.BLASFEO_TARGET;

scen_opts.openmp                = tree_opts.openmp;
scen_opts.linesearch            = tree_opts.linesearch;
scen_opts.reg                   = tree_opts.reg;

scen_opts.CHECK_LAST_ACTIVE_SET = tree_opts.CHECK_LAST_ACTIVE_SET;

%% tree HPMPC options

hpmpc_opts.name                 = 'hpmpc';

hpmpc_opts.maxIter              = 50;
hpmpc_opts.alpha_min            = 1e-8;
hpmpc_opts.mu_tol               = 1e-10;

hpmpc_opts.BLASFEO_LA           = tree_opts.BLASFEO_LA;
hpmpc_opts.BLASFEO_TARGET       = tree_opts.BLASFEO_TARGET;

%% Run algorithms for defined simulations

NSIM = length(NH);
NMOD = length(models);

sim_log_tree_coldstart = cell(NMOD, NSIM);
sim_log_tree_warmstart = cell(NMOD, NSIM);
sim_log_scen_coldstart = cell(NMOD, NSIM);
sim_log_scen_warmstart = cell(NMOD, NSIM);
sim_log_hpmpc          = cell(NMOD, NSIM);

for ii = 1:NMOD
    
    benchmark = models{ii};
    
    for jj = 1:length(NH)
        
        fprintf(fid, 'Running %s simulation %d out of %d with Nh=%d, Nr=%d, m_d=%d\n', ...
            models{ii}, jj, NSIM, NH(jj), NR(jj), MD(jj));
        
        sim_opts.Nh = NH(jj);
        sim_opts.Nr = NR(jj);
        sim_opts.md = MD(jj);
        
        if RUN_TREE
            % run cold-started algorithm
            rng(jj);
            tree_opts.WARMSTART = 0;
            sim_log_tree_coldstart{ii, jj} = treeqp_main(benchmark, sim_opts, tree_opts);
            
            [~, nxb, nub] = number_of_active_constraints(benchmark, sim_log_tree_coldstart{ii, jj}.trajectories);
            fprintf(fid, 'total number of state and input active contsraints: %d and %d\n', sum(nxb), sum(nub));
            
            % run warm-started algorithm
            rng(jj);
            tree_opts.WARMSTART = 1;
            sim_log_tree_warmstart{ii, jj} = treeqp_main(benchmark, sim_opts, tree_opts);
        end
        
        if RUN_SCENARIOS
            % run cold-started algorithm
            rng(jj);
            scen_opts.WARMSTART = 0;
            sim_log_scen_coldstart{ii, jj} = treeqp_main(benchmark, sim_opts, scen_opts);
            
            % run warm-started algorithm
            rng(jj);
            scen_opts.WARMSTART = 1;
            sim_log_scen_warmstart{ii, jj} = treeqp_main(benchmark, sim_opts, scen_opts);
        end
        
        if RUN_HPMPC
            % run cold-started algorithm
            rng(jj);
            sim_log_hpmpc{ii, jj} = treeqp_main(benchmark, sim_opts, hpmpc_opts);
        end
        
        % save current workspace
        eval(['save ' t]);
    end
end

fclose(fid);

if sim_opts.NRUNS < 10
    warning('Small number of runs detected!')
    keyboard
end

%% Store timings, iterations, active-sets

AS_TOL = 1e-4; % tolerance for active constraints

Tcpu   = [];
Titer  = [];
Tas_x  = [];
Tas_u  = [];
Tkkt   = [];
Terr   = []; % TODO: polish

for ii = 1:NMOD
    
    benchmark = models{ii};
    
    for jj = 1:length(NH)
        
        % gather information
        indx    = 0;
        tmpCpu  = nan(sim_opts.nMPC, 2*RUN_TREE + 2*RUN_SCENARIOS + RUN_HPMPC);
        tmpIter = nan(sim_opts.nMPC, 2*RUN_TREE + 2*RUN_SCENARIOS + RUN_HPMPC);
        tmpASx  = nan(sim_opts.nMPC, 2*RUN_TREE + 2*RUN_SCENARIOS + RUN_HPMPC);
        tmpASu  = nan(sim_opts.nMPC, 2*RUN_TREE + 2*RUN_SCENARIOS + RUN_HPMPC);
        tmpKKT  = nan(sim_opts.nMPC, 2*RUN_TREE + 2*RUN_SCENARIOS + RUN_HPMPC);
        tmpErr  = nan(sim_opts.nMPC, 2*RUN_TREE + 2*RUN_SCENARIOS + RUN_HPMPC);
        
        if RUN_TREE
            [~, nASx_tree_cold, nASu_tree_cold] = number_of_active_constraints(benchmark, sim_log_tree_coldstart{ii, jj}.trajectories, AS_TOL);
            [~, nASx_tree_warm, nASu_tree_warm] = number_of_active_constraints(benchmark, sim_log_tree_warmstart{ii, jj}.trajectories, AS_TOL);
            
            % check error
            err1 = max(check_error_in_trajectories(sim_log_tree_coldstart{ii, jj}.trajectories, sim_log_tree_warmstart{ii, jj}.trajectories), [], 2);
            if RUN_SCENARIOS
                err2 = max(check_error_in_trajectories(sim_log_tree_warmstart{ii, jj}.trajectories, sim_log_scen_warmstart{ii, jj}.trajectories), [], 2);
            elseif RUN_HPMPC
                err2 = max(check_error_in_trajectories(sim_log_tree_warmstart{ii, jj}.trajectories, sim_log_hpmpc{ii, jj}.trajectories), [], 2);
            else
                err2 = zeros(sim_opts.nMPC,1);
            end
            
            tmpCpu(:,indx+1:indx+2)  = [sim_log_tree_coldstart{ii, jj}.cpuTime' sim_log_tree_warmstart{ii, jj}.cpuTime'];
            tmpIter(:,indx+1:indx+2) = [sim_log_tree_coldstart{ii, jj}.iter' sim_log_tree_warmstart{ii, jj}.iter'];
            tmpASx(:,indx+1:indx+2)  = [nASx_tree_cold nASx_tree_warm];
            tmpASu(:,indx+1:indx+2)  = [nASu_tree_cold nASu_tree_warm];
            tmpErr(:,indx+1:indx+2)  = [err1 err2];
            tmpKKT(:,indx+1:indx+2)  = [sim_log_tree_coldstart{ii, jj}.kkt' sim_log_tree_warmstart{ii, jj}.kkt'];
            indx                     = indx + 2;
            
        end
        if RUN_SCENARIOS
            [~, nASx_scen_cold, nASu_scen_cold] = number_of_active_constraints(benchmark, sim_log_scen_coldstart{ii, jj}.trajectories, AS_TOL);
            [~, nASx_scen_warm, nASu_scen_warm] = number_of_active_constraints(benchmark, sim_log_scen_warmstart{ii, jj}.trajectories, AS_TOL);
            
            % check error
            err1 = max(check_error_in_trajectories(sim_log_scen_coldstart{ii, jj}.trajectories, sim_log_scen_warmstart{ii, jj}.trajectories), [], 2);
            if RUN_HPMPC
                err2 = max(check_error_in_trajectories(sim_log_scen_warmstart{ii, jj}.trajectories, sim_log_hpmpc{ii, jj}.trajectories), [], 2);
            else
                err2 = zeros(sim_opts.nMPC,1);
            end
            
            tmpCpu(:,indx+1:indx+2)  = [sim_log_scen_coldstart{ii, jj}.cpuTime' sim_log_scen_warmstart{ii, jj}.cpuTime'];
            tmpIter(:,indx+1:indx+2) = [sim_log_scen_coldstart{ii, jj}.iter' sim_log_scen_warmstart{ii, jj}.iter'];
            tmpASx(:,indx+1:indx+2)  = [nASx_scen_cold nASx_scen_warm];
            tmpASu(:,indx+1:indx+2)  = [nASu_scen_cold nASu_scen_warm];
            tmpErr(:,indx+1:indx+2)  = [err1 err2];
            tmpKKT(:,indx+1:indx+2)  = [sim_log_scen_coldstart{ii, jj}.kkt' sim_log_scen_warmstart{ii, jj}.kkt'];
            indx                     = indx + 2;
        end
        if RUN_HPMPC
            [~, nASx_hpmpc, nASu_hpmpc] = number_of_active_constraints(benchmark, sim_log_hpmpc{ii, jj}.trajectories, AS_TOL);
            
            % check error
            if RUN_TREE && RUN_SCENARIOS
                err1 = max(check_error_in_trajectories(sim_log_tree_warmstart{ii, jj}.trajectories, sim_log_hpmpc{ii, jj}.trajectories), [], 2);
            else
                err1 = zeros(sim_opts.nMPC,1);
            end
            
            tmpCpu(:,indx+1)  = sim_log_hpmpc{ii, jj}.cpuTime';
            tmpIter(:,indx+1) = sim_log_hpmpc{ii, jj}.iter';
            tmpASx(:,indx+1)  = nASx_hpmpc;
            tmpASu(:,indx+1)  = nASu_hpmpc;
            tmpKKT(:,indx+1)  = sim_log_hpmpc{ii, jj}.kkt';
            tmpErr(:,indx+1)  = err1;
        end
        
        Tcpu  = [Tcpu;  tmpCpu];  %#ok<AGROW>
        Titer = [Titer; tmpIter]; %#ok<AGROW>
        Tas_x = [Tas_x; tmpASx];  %#ok<AGROW>
        Tas_u = [Tas_u; tmpASu];  %#ok<AGROW>
        Terr  = [Terr; tmpErr];   %#ok<AGROW>
        Tkkt  = [Tkkt; tmpKKT];   %#ok<AGROW>
        
    end
end

%% Check maximum number of iterations and print info

disp('******************************* ITERS *********************************')
disp(' ');

max_iters = nan(1, 2*RUN_TREE + 2*RUN_SCENARIOS + RUN_HPMPC);
indx      = 0;

if RUN_TREE
    max_iters(indx+1:indx+2) = tree_opts.maxIter;
    indx = indx + 2;
end
if RUN_SCENARIOS
    max_iters(indx+1:indx+2) = scen_opts.maxIter;
    indx = indx + 2;
end
if RUN_HPMPC
    max_iters(indx+1) = hpmpc_opts.maxIter;
end

max_iters = repmat(max_iters, size(Tcpu,1), 1);
failed    = sum(Titer >= max_iters);
indx      = 0;

if RUN_TREE
    fprintf("TDUNES (c): %d out of %d instances reached maximum number of iterations\n", failed(indx+1), size(Tcpu, 1));
    fprintf("TDUNES (w): %d out of %d instances reached maximum number of iterations\n", failed(indx+2), size(Tcpu, 1));
    indx = indx + 2;
end
if RUN_SCENARIOS
    fprintf("SDUNES (c): %d out of %d instances reached maximum number of iterations\n", failed(indx+1), size(Tcpu, 1));
    fprintf("SDUNES (w): %d out of %d instances reached maximum number of iterations\n", failed(indx+2), size(Tcpu, 1));
    indx = indx + 2;
end
if RUN_HPMPC
    fprintf("HPMPC     : %d out of %d instances reached maximum number of iterations\n", failed(indx+1), size(Tcpu, 1));
end


disp(' ');
disp('***********************************************************************')

%% Check KKT tolerances and print info

disp('******************************** KKT **********************************')
disp(' ');

KKT_TOLS = {1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3};

for ii = 1:length(KKT_TOLS)
    
    above    = sum(Tkkt > KKT_TOLS{ii});
    indx     = 0;
    
    if RUN_TREE
        fprintf("TDUNES (c): %d out of %d instances exceed %2.2e KKT tolerance\n", above(indx+1), size(Tcpu, 1), KKT_TOLS{ii});
        fprintf("TDUNES (w): %d out of %d instances exceed %2.2e KKT tolerance\n", above(indx+2), size(Tcpu, 1), KKT_TOLS{ii});
        indx = indx + 2;
    end
    if RUN_SCENARIOS
        fprintf("SDUNES (c): %d out of %d instances exceed %2.2e KKT tolerance\n", above(indx+1), size(Tcpu, 1), KKT_TOLS{ii});
        fprintf("SDUNES (w): %d out of %d instances exceed %2.2e KKT tolerance\n", above(indx+2), size(Tcpu, 1), KKT_TOLS{ii});
        indx = indx + 2;
    end
    if RUN_HPMPC
        fprintf("HPMPC     : %d out of %d instances exceed %2.2e KKT tolerance\n", above(indx+1), size(Tcpu, 1), KKT_TOLS{ii});
    end
    disp(' ')
end

disp('***********************************************************************')

%% Find instances with active constraints and print info

disp('************************** ACTIVE CONSTR. *****************************')
disp(' ');

if RUN_TREE && RUN_SCENARIOS
    if any(any(Tas_x(:,1:2) - Tas_x(:,3:4))) || any(any(Tas_u(:,1:2) - Tas_u(:,3:4)))
        warning('detected AS are different')
    end
end

no_x_constr  = sum(Tas_x, 2) == 0;
no_u_constr  = sum(Tas_u, 2) == 0;
unconstr     = no_x_constr & no_u_constr;
Tcpu_constr  = Tcpu(~unconstr, :);
Titer_constr = Titer(~unconstr, :);

fprintf('Percentage of problems with no active constraints:       %2.2f%%\n', mean(unconstr)*100);
fprintf('Percentage of problems with no active state constraints: %2.2f%%\n', mean(no_x_constr)*100);
fprintf('Percentage of problems with no active input constraints: %2.2f%%\n', mean(no_u_constr)*100);

disp(' ')
disp('***********************************************************************')

%% Find worst-case timings

Tcpu_worstcase  = nan(NSIM*NMOD, size(Tcpu,2));
Titer_worstcase = nan(NSIM*NMOD, size(Titer,2));

for ii = 1:NSIM*NMOD
    Tcpu_worstcase(ii,:)  = max(Tcpu((ii-1)*sim_opts.nMPC+1:ii*sim_opts.nMPC,:));
    Titer_worstcase(ii,:) = max(Titer((ii-1)*sim_opts.nMPC+1:ii*sim_opts.nMPC,:));
end

%% Create legend

str = {};

if RUN_TREE
    str{end+1} = 'ALG. DN3 (cold)';
    str{end+1} = 'ALG. DN3 (warm)';
end
if RUN_SCENARIOS
    str{end+1} = 'ALG. DN2 (cold)';
    str{end+1} = 'ALG. DN2 (warm)';
end
if RUN_HPMPC
    str{end+1} = 'ALG. IP1';
end

%% Performance plot

myColorOrder = get(gca, 'ColorOrder');
myColorOrder(1,:) = 0;
set(gca, 'ColorOrder', myColorOrder, 'NextPlot', 'replacechildren');

figure
% unconstrained
if ~DROP_AS_PLOT
    subplot(1,3,1)
else
    subplot(1,2,1)
end

perf(Tcpu)
grid on
set(gca, 'fontsize',18)
title(['All simulations (' num2str(size(Tcpu,1)) ' problems)'], 'interpreter','latex','fontsize',18)
xlim([1 10])
xlabel('$\tau$','interpreter','latex','fontsize',18)
ylabel('$P(r_{\mathrm{p},\, \mathrm{s}} \leq \tau$)','interpreter','latex','fontsize',18)

% constrained
if ~DROP_AS_PLOT
    subplot(1,3,2)
    perf(Tcpu_constr)
    grid on
    set(gca, 'fontsize',18)
    title(['With active constraints (' num2str(size(Tcpu_constr,1)) ' problems)'], 'interpreter','latex','fontsize',18)
    xlim([1 10])
    xlabel('$\tau$','interpreter','latex','fontsize',18)
end

% worst-case
if ~DROP_AS_PLOT
    subplot(1,3,3)
else
    s = subplot(1,2,2);
end

perf(Tcpu_worstcase)
grid on
set(gca, 'fontsize',18)
title(['Worst-case (' num2str(size(Tcpu_worstcase,1)) ' problems)'], 'interpreter','latex','fontsize',18)
xlim([1 10])
xlabel('$\tau$','interpreter','latex','fontsize',18)
l = legend(str);
l.Interpreter = 'latex';
l.Location = 'southeast';

if DROP_AS_PLOT
    f = gcf;
    f.Position = [100 300 1500 600];
    s.Position(1) = s.Position(1) - 0.06;
end
% export
if EXPORT_FIG
    keyboard % manually changed colors and linewidth for paper at this point
    fpath = './';
    exportfig([fpath t '.pdf'])
end

%% historgram

figure

MODE = 3;
FS = 20;

if RUN_TREE == 0 || RUN_SCENARIOS == 0
    error('data not available')
end

if MODE == 1
    tot = size(Titer,1);
    iters_tree_cold = Titer(:,1);
    iters_scen_cold = Titer(:,3);
    iters_tree_warm = Titer(:,2);
    iters_scen_warm = Titer(:,4);
elseif MODE == 2
    tot = size(Titer_constr,1);
    iters_tree_cold = Titer_constr(:,1);
    iters_scen_cold = Titer_constr(:,3);
    iters_tree_warm = Titer_constr(:,2);
    iters_scen_warm = Titer_constr(:,4);
elseif MODE == 3
    tot = size(Titer_worstcase,1);
    iters_tree_cold = Titer_worstcase(:,1);
    iters_scen_cold = Titer_worstcase(:,3);
    iters_tree_warm = Titer_worstcase(:,2);
    iters_scen_warm = Titer_worstcase(:,4);
end

%bin_param = 50;
bin_param = -20:1:20;

subplot(1,2,1)
histogram(iters_scen_warm - iters_tree_warm, bin_param, 'Normalization', 'probability');
title('Warm-start', 'interpreter','latex', 'fontsize',FS);
xlabel('Difference in number of iterations', 'interpreter','latex', 'fontsize',FS)
ylabel('Probability', 'interpreter','latex', 'fontsize',FS)
grid on
set(gca, 'fontsize', FS)
ylim([0 1])
xlim([-10 10])

s = subplot(1,2,2);
histogram(iters_scen_cold - iters_tree_cold, bin_param, 'Normalization', 'probability');
xlabel('Difference in number of iterations', 'interpreter','latex', 'fontsize',FS)
title('Cold-start', 'interpreter','latex', 'fontsize',FS);
grid on
set(gca, 'fontsize', FS)
ylim([0 1])
xlim([-10 10])

if EXPORT_FIG
    
    keyboard % pause here to export correctly
    
    f = gcf;
    f.Position = [100 300 1500 400];
    s.Position(1) = s.Position(1) - 0.06;

    fpath = './';    
    exportfig([fpath t '_histogram.pdf'])
end

%% save data

% save final workspace
eval(['save ' t '_full']);
clear sim_log_hpmpc sim_log_scen_coldstart sim_log_scen_warmstart sim_log_tree_coldstart sim_log_tree_warmstart
eval(['save ' t '_reduced']);
delete([t '.mat'])
