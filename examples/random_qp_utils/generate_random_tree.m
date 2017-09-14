function [ agents ] = generate_random_tree( nc, nx, nu )

% GENERATE_RANDOM_TREE Generate a tree with structure given by nc and 
%   dimensions given by nr, nx, nu.
%
%   nc: number of children of node ( 1 x N ) 
%   nr: realization index of _NEXT_ node ( 1 x N - 1 ) ----- NOT USED ATM
%   nx: number of states ( 1 x N )
%   nu: number of controls ( 1 x N )

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
    
    agents(ii).Q    = diag(10*abs(rand(nx(ii),1)));
    agents(ii).R    = diag(1*abs(rand(nu(ii),1)));
    agents(ii).q    = rand(nx(ii),1);
    agents(ii).r    = rand(nu(ii),1);
end

end
