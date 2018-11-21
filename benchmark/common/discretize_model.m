function [Ad, Bd] = discretize_model(A, B, Ts)

% DISCRETIZE_MODEL use matrix exponentials to calculate discrete-time model
%                  from continous-time linearization

[nx, nu] = size(B);

M  = expm([Ts*A, Ts*B; zeros(nu, nx+nu)]);
Ad = M(1:nx,1:nx);
Bd = M(1:nx,nx+1:end);

end

