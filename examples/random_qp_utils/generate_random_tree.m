function [ agents ] = generate_random_tree( nc, nx, nu, CLIPPING )

% GENERATE_RANDOM_TREE Generate a tree with structure given by nc and 
%   dimensions given by nr, nx, nu.
%
%   nc:         number of children of node ( 1 x N ) 
%   nx:         number of states ( 1 x N )
%   nu:         number of controls ( 1 x N )
%   CLIPPING:   when true (default), Q,R are diagonal and S is zero

if nargin == 3
    CLIPPING = true;
end

% initialize all nodes to unassigned
for ii = 1:length(nc)
    agents(ii).stage = -1;
    agents(ii).idx = NaN;
    agents(ii).dad = NaN;
    agents(ii).idxkid = NaN;
    agents(ii).nkids = NaN;
end

% root
agents(1).idx   = 1;
agents(1).dad   = -1;
agents(1).stage = 0;
agents(1).idxkid = 0;

% initialize tree
for ii = 1: length(nc)
    %disp(['initializing node ' num2str(ii)]);
    
    agents(ii).nkids = nc(ii);
    
    % identify where children nodes (with unasigned stage) are
    for jj = ii:length(nc)
        if agents(jj).stage == -1
            break;
        end
    end
    
    for kk = jj:jj + nc(ii) - 1 
        agents(kk).idx    = kk; 
        agents(kk).dad    = ii;
        agents(kk).stage  = agents(ii).stage + 1;
        agents(kk).idxkid = kk - jj;
    end
    
end

for ii = 1:length(nc)
    iidad = agents(ii).dad;
    
    if ii > 1
        agents(ii).A    = rand(nx(ii), nx(iidad));
        agents(ii).B    = rand(nx(ii), nu(iidad));
        agents(ii).b    = rand(nx(ii), 1);
    end
    
    if CLIPPING
        agents(ii).Q    = diag(10*abs(rand(nx(ii),1)));
        agents(ii).R    = diag(1*abs(rand(nu(ii),1)));
        agents(ii).S    = zeros(nu(ii), nx(ii));
    else
        agents(ii).Q    = rand(nx(ii)) + diag(10*abs(rand(nx(ii),1)));
        agents(ii).Q    = (agents(ii).Q + agents(ii).Q')./2
        agents(ii).R    = rand(nu(ii)) + diag(2*abs(rand(nu(ii),1)));
        agents(ii).R    = (agents(ii).R + agents(ii).R')./2
        agents(ii).S    = rand(nu(ii), nx(ii));
    end
    
    H = [agents(ii).Q agents(ii).S'; agents(ii).S agents(ii).R];
    if any(eig(H)) <= 0
        error(['Hessian of node ' double2str(ii) ' not positive definite!']);
    end
    
    agents(ii).q    = rand(nx(ii),1);
    agents(ii).r    = rand(nu(ii),1);
end

end
