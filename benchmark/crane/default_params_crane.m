function param = default_params_crane()

% DEFAULT_PARAMS_CRANE Return structure with default parameters for crane
%                      example.

param.b    = 0.2;  % friction coefficient (uncertain parameter)
param.bmin = 0.1;  % minimum possible value for friction coefficient
param.bmax = 0.3;  % maximum possible value for friction coefficient
param.g    = 9.81; % gravitational acceleration
param.inf  = 1e8;  % infinity value for non-constrained states/controls
param.Ts   = 0.2;  % sampling time

end


