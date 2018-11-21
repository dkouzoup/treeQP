function [ trajectories ] = build_trajectories_from_trees(agents, Nh, Nr )

% BUILD_TRAJECTORIES_FROM_TREES Read current scenario data and form
%                               state and input trajectories

md = length(agents(1).child);
Nscenarios = md^Nr;

[nx, nu] = size(agents(2).dyn.B);

trajectories(Nscenarios).x = [];
trajectories(Nscenarios).u = [];
for ii = 1:Nscenarios
    
    x = zeros(nx,Nh+1);
    u = zeros(nu,Nh);
    
    x(:,end) = agents(end-ii+1).xit;
    
    kk = agents(end-ii+1).parent;
    for jj = 1:Nh
        x(:,end-jj)   = agents(kk).xit;
        u(:,end-jj+1) = agents(kk).uit;
        kk = agents(kk).parent;
    end
    
    trajectories(Nscenarios-ii+1).x = x; % reverse order to match results in main_scenarios.m
    trajectories(Nscenarios-ii+1).u = u;
    
end




end

