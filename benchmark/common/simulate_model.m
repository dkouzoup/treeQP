function [ fun ] = simulate_model(xlin, ulin, Ts, dynamics, params)

% SIMULATE_MODEL Create a function handle that returns x_next based on
%                current states, controls and linearization point. 

fun = @(x, u) integrate_RK4(x + xlin, u + ulin, Ts, dynamics, params);


end

