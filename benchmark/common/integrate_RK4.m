function x_next = integrate_RK4(x, u, h, rhs, param)

% INTEGRATE_RK4 Runge-Kutta integrator of order 4.
    
    k1 = rhs(x, u, param);
    k2 = rhs(x+h/2*k1, u, param);
    k3 = rhs(x+h/2*k2, u, param);
    k4 = rhs(x+h*k3, u, param);
    
    x_next = x + h/6*(k1+2*k2+2*k3+k4);
    
end
