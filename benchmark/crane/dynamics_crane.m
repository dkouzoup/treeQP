function [ dx ] = dynamics_crane(x, u, param)

% DYNAMICS_CRANE crane dynamics used in RK4 integrator and MPC controller.

b = param.b;
g = param.g;

% extract states and controls
% p   = x(1);
v     = x(2);
phi   = x(3);
omega = x(4);
a     = u;

% build RHS
dx  = [v; ...
       a; ...
       omega; ...
       -g*sin(phi)-a*cos(phi)-b*omega];

end
