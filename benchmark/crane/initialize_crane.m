function [dynamics, weights, constraints, nominal] = initialize_crane(NSIM, params_mpc, param_sim)

% INITIALIZE_CRANE build crane model and reference signals.
%
% NSIM:       number of time steps for reference signals (should be bigger 
%             or equal to number of problems in closed-loop simulation.
%
% params_mpc: cell of md structures with parameters of MPC controller.
% 
% params_sim: structure withe parameters of (nonlinear) simulator.

% initialization

if nargin < 2 || isempty(params_mpc)
    nreal         = 1;
    params_mpc    = cell(1);
    params_mpc{1} = default_params_crane();
else
    if ~iscell(params_mpc)
        nreal         = 1;
        params_mpc    = cell(1);
        params_mpc{1} = params_mpc;
    else
        nreal = length(params_mpc);
    end
end

if nargin < 3 || isempty(param_sim)
    param_sim = default_params_crane();
end

% dimensions
nx = 4;
nu = 1;
Ts = params_mpc{1}.Ts;

% constraints
constraints.x0    = [0.0; 0; 0; 0];
constraints.xmin  = [-param_sim.inf; -0.2; -param_sim.inf; -0.4];
constraints.xmax  = -constraints.xmin;
constraints.umin  = -0.5*ones(nu,1);
constraints.umax  = +0.5*ones(nu,1);
constraints.xNmin = constraints.xmin;
constraints.xNmax = constraints.xmax;

% objective (diagonal weights)
weights.Q = [10; 1; 1; 1];
weights.R = 0.1;
weights.P = weights.Q;

% reference signals to generate q and r online
weights.xref = [repmat([0.2; 0.0; 0.0; 0.0], 1, ceil(NSIM/2)) repmat([-0.2; 0.0; 0.0; 0.0], 1, floor(NSIM/2))];          
weights.uref = zeros(nu, NSIM);

% dynamics (for different set of parameters)
xlin = zeros(nx, 1);
ulin = zeros(nu, 1);

dynamics = struct();

for ii = 1:nreal
    
    [A, B] = linearize_model(xlin, ulin, @dynamics_crane, params_mpc{ii});
    
    [dynamics(ii).A, dynamics(ii).B] = discretize_model(A, B, Ts); 
    
    dynamics(ii).c = zeros(nx,1);
    
end

nominal.simulate = simulate_model(xlin, ulin, Ts, @dynamics_crane, param_sim);

end

