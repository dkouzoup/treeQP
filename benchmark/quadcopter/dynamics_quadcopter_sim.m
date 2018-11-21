function [ dx ] = dynamics_quadcopter_sim( x, u, par )

% DYNAMICS_QUADCOPTER_SIM quadcopter dynamics used in RK4 integrator.

rho = par.rho;
A   = par.A;
Cl  = par.Cl;
Cd  = par.Cd;
m   = par.m;
g   = par.g;
L   = par.L;
L2  = par.L2; 
J1  = par.J1;
J2  = par.J2;
J3  = par.J3;

q0  = x(1);
q1  = x(2);
q2  = x(3);
q3  = x(4);
q   = [q0; q1; q2; q3];

Omega1 = x(5);
Omega2 = x(6);
Omega3 = x(7);
Omega  = [Omega1; Omega2; Omega3];

alpha  = 0.0; % Baumgartner stabilization

W1 = u(1);
W2 = u(2);
W3 = u(3);
W4 = u(4);

E = [-q1, -q2, -q3; ...
      q0, -q3,  q2; ...
      q3,  q0, -q1; ...
     -q2,  q1,  q0];

dq = 1/2*E*Omega - alpha*q*(q0*q0 + q1*q1 + q2*q2 + q3*q3 - 1)/(q0*q0 + q1*q1 + q2*q2 + q3*q3);
    
dx = [dq;
      (-J3*Omega2*Omega3 + J2*Omega2*Omega3 + (A*Cl*L*rho*(W2*W2 - W4*W4))/2)/J1;
      (J3*Omega1*Omega3 - J1*Omega1*Omega3 + (A*Cl*L*rho*(W3*W3 - W1*W1))/2)/J2;
      (-J2*Omega1*Omega2 + J1*Omega1*Omega2 + (A*Cd*L2*rho*(W1*W1 - W2*W2 + W3*W3 - W4*W4))/2)/J3];
end