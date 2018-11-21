function [ dynamics, weights, constraints, nominal ] = import_benchmark(benchmark, md, NSIM)

% IMPORT_BENCHMARK Load dynamics weights and constraints for the selected
%                  benchmark model.

addpath('common')

% check for casadi
if strcmp(benchmark, 'quadcopter') || strcmp(benchmark, 'crane')
    try
        import casadi.*
        dummy = MX.sym('dummy', 1,1); %#ok<NASGU>
    catch
        error('Casadi not found in your matlab path (required for nonlinear benchmark examples)')
    end
 
end
    
% TODO: add spring-mass benchmark from previous paper
if contains(benchmark, 'linear_chain')
    
    addpath('linear_chain')
    
    % extract params from name
    nm = extract_param_value(benchmark, 'nm');
    nu = extract_param_value(benchmark, 'nu');
    
    % set up parameters for simulator
    param_sim  = default_params_linear_chain(nm, nu);
    param_sim.k = param_sim.kmin + (param_sim.kmax-param_sim.kmin).*rand;
    
    % choose values for uncertainty in MPC controller
    k_mpc = linspace(param_sim.kmin, param_sim.kmax, md);
    
    params = cell(1, md);
    for ii = 1:md
        params{ii}   = default_params_linear_chain(nm, nu);
        params{ii}.k = k_mpc(ii);
    end
    
    [dynamics, weights, constraints, nominal] = initialize_linear_chain(NSIM, params, param_sim);
    
elseif strcmp(benchmark, 'quadcopter')
    
    addpath('quadcopter')
    
    % set up parameters for simulator
    param_sim    = default_params_quadcopter();
    param_sim.m = param_sim.mmin + (param_sim.mmax-param_sim.mmin).*rand;
    
    % choose values for uncertainty in MPC controller
    m_mpc = linspace(param_sim.mmin, param_sim.mmax, md);
    
    params = cell(1, md);
    for ii = 1:md
        params{ii}   = default_params_quadcopter();
        params{ii}.m = m_mpc(ii);
    end
    
    [dynamics, weights, constraints, nominal] = initialize_quadcopter(NSIM, params, param_sim);
    
elseif strcmp(benchmark, 'crane')
    
    addpath('crane')
    
    % set up parameters for simulator
    param_sim   = default_params_crane();
    param_sim.b = param_sim.bmin + (param_sim.bmax-param_sim.bmin).*rand;
    
    % choose values for uncertainty in MPC controller
    b_mpc = linspace(param_sim.bmin, param_sim.bmax, md);
    
    params = cell(1, md);
    for ii = 1:md
        params{ii}   = default_params_crane();
        params{ii}.b = b_mpc(ii);
    end
    
    [dynamics, weights, constraints, nominal] = initialize_crane(NSIM, params, param_sim);
    
else
    error('Unknown model specified')
    
end

end

