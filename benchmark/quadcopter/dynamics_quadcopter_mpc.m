function [ dx ] = dynamics_quadcopter_mpc(x, u, par)

% DYNAMICS_QUADCOPTER_MPC quadcopter dynamics used in RK4 integrator.

rho = par.rho;
A   = par.A;
Cl  = par.Cl;
Cd  = par.Cd;
L   = par.L;
L2  = par.L2;
J1  = par.J1;
J2  = par.J2;
J3  = par.J3;

q2  = x(1);
q3  = x(2);
q4  = x(3);
q1  = sqrt(1 - q2^2 - q3^2 - q4^2);

Omega1 = x(4);
Omega2 = x(5);
Omega3 = x(6);

W1 = u(1);
W2 = u(2);
W3 = u(3);
W4 = u(4);

quat_dyn = 1/2*[ q1, -q4,  q3;...
                 q4,  q1, -q2;...
                -q3,  q2,  q1]*[Omega1; Omega2; Omega3];

dx = [quat_dyn;
      (-J3*Omega2*Omega3 + J2*Omega2*Omega3 + (A*Cl*L*rho*(W2*W2 - W4*W4))/2)/J1;
      (J3*Omega1*Omega3 - J1*Omega1*Omega3 + (A*Cl*L*rho*(W3*W3 - W1*W1))/2)/J2;
      (-J2*Omega1*Omega2 + J1*Omega1*Omega2 + (A*Cd*L2*rho*(W1*W1 - W2*W2 + W3*W3 - W4*W4))/2)/J3];    
end
