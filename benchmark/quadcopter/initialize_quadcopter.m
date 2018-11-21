function [dynamics, weights, constraints, nominal] = initialize_quadcopter(NSIM, params_mpc, param_sim)

% INITIALIZE_QUADCOPTER build quadcopter model and reference signals.
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
    params_mpc{1} = default_params_quadcopter();
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
    param_sim = default_params_quadcopter();
end

% dimensions
nx = 6;
nu = 4;
Ts = params_mpc{1}.Ts;

% constraints
q0 = myeul2quat(0, 0, 0);
x0 = [q0; 0; 0; 0 ]; % 1st state is eliminated based on quartenion condition

delta_u_lim       = 4;
constraints.x0    = x0(2:end);
constraints.xmin  = [-param_sim.inf*ones(3,1); -1*ones(3,1)];
constraints.xmax  = -constraints.xmin;
constraints.umin  = -delta_u_lim*ones(nu,1);
constraints.umax  = +delta_u_lim*ones(nu,1);
constraints.xNmin = -param_sim.inf*ones(nx,1);
constraints.xNmax = +param_sim.inf*ones(nx,1);

% objective (diagonal weights)
weights.Q = [500; 500; 500; 0.001; 0.001; 0.001];
weights.R = [0.001; 0.001; 0.001; 0.001];
weights.P = weights.Q;

% reference signals to generate q and r online
REF_PERIOD = floor(NSIM/4);
REF_ANGLE  = -1*pi/2/9*5*0.1;
LAMBDA     = 0.3;
ref        = zeros(3, NSIM);
ref_prev   = ref(:,1);
q_ref      = zeros(nx+1, NSIM);
x_ref      = zeros(nx, NSIM);

for i = 1:NSIM

    ref_phase = mod(floor((i-1)/REF_PERIOD),3);

    if (ref_phase == 0)
        ref(1,i) = ref_prev(1) - LAMBDA*(ref_prev(1) + REF_ANGLE);
        ref(2,i) = ref_prev(2) - LAMBDA*(ref_prev(2) - REF_ANGLE);
        ref_prev = ref(:,i);
    elseif (ref_phase == 1)
        ref(1,i) = ref_prev(1) - LAMBDA*(ref_prev(1) - REF_ANGLE);
        ref(2,i) = ref_prev(2) - LAMBDA*(ref_prev(2) - REF_ANGLE);
        ref_prev = ref(:,i);
    else
        ref(1,i) = ref_prev(1) - LAMBDA*(ref_prev(1) - REF_ANGLE);
        ref(2,i) = ref_prev(2) - LAMBDA*(ref_prev(2) + REF_ANGLE);
        ref_prev = ref(:,i);
    end

    q_ref(:,i) = [myeul2quat(ref(1, i), ref(2, i), ref(3, i)); zeros(3,1)];
    x_ref(:,i) = q_ref(2:end, i);
end

weights.eulref = ref;
weights.xref   = x_ref;
weights.uref   = zeros(nu, NSIM);

% dynamics (for different set of parameters)
xlin = zeros(nx,1);

dynamics = struct();

for ii = 1:nreal

    par = params_mpc{ii};

    omega_hover = sqrt(2*par.m*par.g/(par.A*par.Cl*par.rho)/4);
    ulin        = omega_hover*ones(nu, 1);

    [A, B] = linearize_model(xlin, ulin, @dynamics_quadcopter_mpc, par);

    [dynamics(ii).A, dynamics(ii).B] = discretize_model(A, B, Ts);

    dynamics(ii).c = zeros(nx,1);

end

omega_hover = sqrt(2*param_sim.m*param_sim.g/(param_sim.A*param_sim.Cl*param_sim.rho)/4);
ulin        = omega_hover*ones(nu, 1);
nominal.simulate = simulate_model([0; xlin], ulin, Ts, @dynamics_quadcopter_sim, param_sim);

end

