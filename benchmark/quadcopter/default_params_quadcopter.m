function param = default_params_quadcopter()

% DEFAULT_PARAMS_QUADCOPTER Return structure with default parameters for 
%                           quadcopter example.


param.rho   = 1.23;           % air density
param.A     = 0.1;            % propeller area
param.Cl    = 0.25;           % lift coefficient
param.Cd    = 0.3*param.Cl;   % drag coefficient
param.m     = 10;             % quadrotor mass (uncertain parameter)
param.mmin  = 8;              % minimum possible value for quadrotor mass
param.mmax  = 12;             % maximum possible value for quadrotor mass
param.g     = 9.81;           % gravitational acceleration
param.L     = 0.5;
param.L2    = 1;
param.J1    = 0.25;           % moment of inertia
param.J2    = 0.25;           % J2 = J1
param.J3    = 1;              % J3 = 4*J1
param.inf   = 1e8;            % infinity value for non-constrained states/controls
param.Ts    = 0.05;           % sampling time

end

