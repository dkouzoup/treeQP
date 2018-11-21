function param = default_params_linear_chain(nm, nu)

% DEFAULT_PARAMS_LINEAR_CHAIN Return structure with default parameters for 
%                             linear chain of masses example.

if nargin < 2
    nu = 3;
end
if nargin < 1
    nm = 4;
end
   
if nu >= nm
    error('actuated masses cannot be more than nm-1');
end

param.nm   = nm;    % number of masses
param.nu   = nu;    % number of controls (nu < nm)
param.inf  = 1e8;   % infinity value for non-constrained states/controls
param.Ts   = 0.05;  % sampling time
param.k    = 6;     % spring constant (uncertain parameter)
param.kmin = 4;     % minimum possible value for spring  constant
param.kmax = 8;     % maximum possible value for spring  constant

end

