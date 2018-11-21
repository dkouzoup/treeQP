function [dynamics, weights, constraints, nominal] = initialize_linear_chain(NSIM, params_mpc, param_sim)

% INITIALIZE_LINEAR_CHAIN build linear chain model.
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
    params_mpc{1} = default_params_linear_chain();
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
    param_sim = default_params_linear_chain();
end

% dimensions
% TODO: add checks to make sure those are the same for all param structs
nm = params_mpc{1}.nm;
nx = 2*nm;
nu = params_mpc{1}.nu;
Ts = params_mpc{1}.Ts;

% constraints
x0      = zeros(nx, 1);
ind     = nm+nu+1; % index of an uncontrolled mass (to give initial velocity)
x0(ind) = 2.0;
pos_ub  = +2.0*ones(nx/2,1);
pos_lb  = -2.0*ones(nx/2,1);
vel_ub  = +2.0*ones(nx/2,1);
vel_lb  = -2.0*ones(nx/2,1);
u_ub    = +2.0;
u_lb    = -2.0;

constraints.x0    = x0;
constraints.xmin  = [pos_lb; vel_lb];
constraints.xmax  = [pos_ub; vel_ub];
constraints.umin  = u_lb*ones(nu,1);
constraints.umax  = u_ub*ones(nu,1);
constraints.xNmin = constraints.xmin;
constraints.xNmax = constraints.xmax;

% objective (diagonal weights)
weights.R = ones(nu,1);
weights.Q = 10*ones(nx,1);
weights.P = weights.Q;

% reference signals to generate q and r online
weights.xref = zeros(nx, NSIM);
weights.uref = zeros(nu, NSIM);

% dynamics (for different set of parameters)
dynamics = struct();

T = diag(-2*ones(nx/2,1))+diag(ones(nx/2-1,1),-1)+diag(ones(nx/2-1,1),1); % matrix describing the effect of the spings

for ii = 1:nreal
    
    k = params_mpc{ii}.k;
    A = [zeros(nx/2), eye(nx/2); k*T, zeros(nx/2)];
    B = [zeros(nx/2,nu); eye(nu); zeros(nx/2-nu, nu)];
    
    [dynamics(ii).A, dynamics(ii).B] = discretize_model(A, B, Ts); 
    
    dynamics(ii).c = zeros(nx,1);
    
end

ksim = param_sim.k;
Asim = [zeros(nx/2), eye(nx/2); ksim*T, zeros(nx/2)];
Bsim = [zeros(nx/2,nu); eye(nu); zeros(nx/2-nu, nu)];
[Asim, Bsim] =  discretize_model(Asim, Bsim, Ts);

nominal.simulate = @(x, u) Asim*x + Bsim*u;

end

