function [A, B] = linearize_model(xlin, ulin, rhs, param)

% LINEARIZE_MODEL linearize nonlinear model around given point.

import casadi.*

% build RHS as casadi expression
nx   = length(xlin);
nu   = length(ulin);
x    = MX.sym('x',nx,1);
u    = MX.sym('u',nu,1);
xdot = rhs(x, u, param);

% evaluate A and B at linearization pointå
A_expr = jacobian(xdot, x);
A_fun  = Function('A', {x, u}, {A_expr});
B_expr = jacobian(xdot, u);
B_fun  = Function('B', {x, u}, {B_expr});
A      = full(A_fun(xlin,ulin));
B      = full(B_fun(xlin, ulin));

end

