function [trajectories, nASx, nASu] = number_of_active_constraints(benchmark, trajectories, TOL )

% NUMBER_OF_ACTIVE_CONSTRAINTS Calculate number of active state and control
%   constraints at optimal solution
    
if nargin < 3
    TOL = 1e-10;
end

if iscell(trajectories)
    NSIM = length(trajectories);
else 
    NSIM = 1;
    tmp  = trajectories;
    clear trajectories
    trajectories{1} = tmp;
end

NMPC  = length(trajectories);
NSCEN = length(trajectories{1});
NU    = size(trajectories{1}(1).u, 1);
N     = size(trajectories{1}(1).u,2);
NX    = size(trajectories{1}(1).x,1);

[~, ~, constraints, ~] = import_benchmark(benchmark, 1, length(trajectories));

XMAX = [repmat(constraints.xmax,1,N) constraints.xNmax];
XMIN = [repmat(constraints.xmin,1,N) constraints.xNmin];
UMAX = repmat(constraints.umax,1,N);
UMIN = repmat(constraints.umin,1,N);
nASx = zeros(NSIM, 1);
nASu = zeros(NSIM, 1);

for ii = 1:NSIM  
    for jj = 1:NSCEN
        trajectories{ii}(jj).xAS = (abs(XMAX-trajectories{ii}(jj).x) <= TOL) | ...
            (abs(trajectories{ii}(jj).x-XMIN) <= TOL);
        trajectories{ii}(jj).uAS = (abs(UMAX-trajectories{ii}(jj).u) <= TOL) | ...
            (abs(trajectories{ii}(jj).u-UMIN) <= TOL);
    end
    
    tmpxAS   = horzcat(trajectories{ii}.xAS);
    tmpuAS   = horzcat(trajectories{ii}.uAS);
    nASx(ii) = sum(sum(tmpxAS));
    nASu(ii) = sum(sum(tmpuAS));
end

end

